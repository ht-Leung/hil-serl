#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
import signal
import sys
import gc

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    client = None
    pbar = None
    
    try:
        if FLAGS.eval_checkpoint_step:
            success_counter = 0
            time_list = []

            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=FLAGS.eval_checkpoint_step,
            )
            agent = agent.replace(state=ckpt)

            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                done = False
                start_time = time.time()
                while not done:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=False,
                        seed=key
                    )
                    actions = np.asarray(jax.device_get(actions))

                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                    if done:
                        if reward:
                            dt = time.time() - start_time
                            time_list.append(dt)
                            print(dt)

                        success_counter += reward
                        print(reward)
                        print(f"{success_counter}/{episode + 1}")

            print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
            print(f"average time: {np.mean(time_list)}")
            return  # after done eval, return and exit
        
        start_step = (
            int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
            if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
            else 0
        )

        datastore_dict = {
            "actor_env": data_store,
            "actor_env_intvn": intvn_data_store,
        }

        client = TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(),
            data_stores=datastore_dict,
            wait_for_server=True,
            timeout_ms=3000,
        )

        # Function to update the agent with new params
        def update_params(params):
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))

        client.recv_network_callback(update_params)

        transitions = []
        demo_transitions = []

        obs, _ = env.reset()
        done = False

        # training loop
        timer = Timer()
        running_return = 0.0
        already_intervened = False
        intervention_count = 0
        intervention_steps = 0
        
        # Metrics tracking for Figure 4
        training_start_time = time.time()  # Track wall-clock time from training start
        episode_count = 0
        success_history = []  # Store last 20 episodes
        autonomous_success_history = []  # Store last 20 episodes for autonomous success
        assisted_success_history = []  # Store last 20 episodes for assisted success
        cycle_time_history = []  # Store last 20 episodes - all episodes
        autonomous_cycle_time_history = []  # Store last 20 episodes - autonomous only
        assisted_cycle_time_history = []  # Store last 20 episodes - with intervention  
        intervention_rate_history = []  # Store last 20 episodes
        episode_start_time = time.time()
        episode_steps = 0

        pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
        for step in pbar:
            timer.tick("total")

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

            # Step environment
            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(actions)
                episode_steps += 1
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                # override the action with the intervention action
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False

                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                if 'grasp_penalty' in info:
                    transition['grasp_penalty']= info['grasp_penalty']
                data_store.insert(transition)
                transitions.append(copy.deepcopy(transition))
                if already_intervened:
                    intvn_data_store.insert(transition)
                    demo_transitions.append(copy.deepcopy(transition))

                obs = next_obs
                if done or truncated:
                    episode_count += 1
                    
                    # Calculate metrics for this episode
                    cycle_time = time.time() - episode_start_time
                    intervention_rate = intervention_steps / max(episode_steps, 1)
                    # Check success from episode info - RecordEpisodeStatistics stores cumulative reward in "r"
                    episode_reward = info.get("episode", {}).get("r", 0)
                    success = episode_reward > 0
                    
                    # Calculate different types of success
                    autonomous_success = success and (intervention_steps == 0)  # Completely autonomous success
                    assisted_success = success and (intervention_steps > 0)      # Success with human assistance
                    
                    # Update history (keep last 20 episodes)
                    success_history.append(float(success))
                    autonomous_success_history.append(float(autonomous_success))
                    assisted_success_history.append(float(assisted_success))
                    cycle_time_history.append(cycle_time)
                    intervention_rate_history.append(intervention_rate)
                    
                    # Record cycle time by intervention type
                    if intervention_steps == 0:
                        autonomous_cycle_time_history.append(cycle_time)
                    else:
                        assisted_cycle_time_history.append(cycle_time)
                    
                    # Maintain history size (keep last 20 episodes)
                    if len(success_history) > 20:
                        success_history.pop(0)
                        autonomous_success_history.pop(0)
                        assisted_success_history.pop(0)
                        cycle_time_history.pop(0)
                        intervention_rate_history.pop(0)
                    
                    if len(autonomous_cycle_time_history) > 20:
                        autonomous_cycle_time_history.pop(0)
                    if len(assisted_cycle_time_history) > 20:
                        assisted_cycle_time_history.pop(0)
                    
                    # Calculate running averages
                    avg_success_rate = np.mean(success_history) if success_history else 0.0
                    avg_autonomous_success_rate = np.mean(autonomous_success_history) if autonomous_success_history else 0.0
                    avg_assisted_success_rate = np.mean(assisted_success_history) if assisted_success_history else 0.0
                    avg_cycle_time = np.mean(cycle_time_history) if cycle_time_history else 0.0
                    avg_autonomous_cycle_time = np.mean(autonomous_cycle_time_history) if autonomous_cycle_time_history else 0.0
                    avg_assisted_cycle_time = np.mean(assisted_cycle_time_history) if assisted_cycle_time_history else 0.0
                    avg_intervention_rate = np.mean(intervention_rate_history) if intervention_rate_history else 0.0
                    
                    # Calculate training time in minutes
                    training_time_minutes = (time.time() - training_start_time) / 60.0
                    
                    # Add metrics to info
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    info["episode"]["intervention_rate"] = intervention_rate
                    info["episode"]["cycle_time"] = cycle_time
                    info["episode"]["success"] = float(success)
                    info["episode"]["autonomous_success"] = float(autonomous_success)
                    info["episode"]["assisted_success"] = float(assisted_success)
                    
                    # Add Figure 4 metrics with training time
                    info["figure4_metrics"] = {
                        "success_rate_avg20": avg_success_rate,
                        "autonomous_success_rate_avg20": avg_autonomous_success_rate,
                        "assisted_success_rate_avg20": avg_assisted_success_rate,
                        "cycle_time_avg20": avg_cycle_time,
                        "autonomous_cycle_time_avg20": avg_autonomous_cycle_time,
                        "assisted_cycle_time_avg20": avg_assisted_cycle_time,
                        "intervention_rate_avg20": avg_intervention_rate,
                        "training_time_minutes": training_time_minutes,
                        "episode_count": episode_count,
                    }
                    
                    stats = {"environment": info}  # send stats to the learner to log
                    client.request("send-stats", stats)
                    pbar.set_description(f"return: {running_return:.1f} | auto: {avg_autonomous_success_rate:.0%} | total: {avg_success_rate:.0%} | int: {avg_intervention_rate:.0%} | t: {training_time_minutes:.1f}m")
                    
                    # Reset for next episode
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    episode_steps = 0
                    episode_start_time = time.time()
                    already_intervened = False
                    client.update()
                    obs, _ = env.reset()

            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
                # dump to pickle file
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                if not os.path.exists(buffer_path):
                    os.makedirs(buffer_path)
                if not os.path.exists(demo_buffer_path):
                    os.makedirs(demo_buffer_path)
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
                    transitions = []
                with open(
                    os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
                ) as f:
                    pkl.dump(demo_transitions, f)
                    demo_transitions = []

            timer.tock("total")

            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)
    
    except KeyboardInterrupt:
        print("\n[INFO] Actor interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Actor unexpected error: {e}")
        raise
    finally:
        print("\n[INFO] Actor cleanup starting...")
        
        # Close progress bar
        if pbar is not None:
            try:
                pbar.close()
            except Exception as e:
                print(f"[WARNING] Error closing progress bar: {e}")
        
        # Stop client
        if client is not None:
            try:
                client.stop()
                print("[INFO] TrainerClient stopped")
            except Exception as e:
                print(f"[WARNING] Error stopping client: {e}")
        
        # Close environment
        if env is not None:
            try:
                if hasattr(env, 'close'):
                    env.close()
                    print("[INFO] Environment closed")
            except Exception as e:
                print(f"[WARNING] Error closing environment: {e}")
        
        print("[INFO] Actor cleanup completed")


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    server = None
    pbar = None
    
    try:
        start_step = (
            int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
            + 1
            if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
            else 0
        )
        step = start_step

        def stats_callback(type: str, payload: dict) -> dict:
            """Callback for when server receives stats request."""
            assert type == "send-stats", f"Invalid request type: {type}"
            if wandb_logger is not None:
                # Check if payload contains figure4_metrics with training time
                if ("environment" in payload and 
                    "figure4_metrics" in payload["environment"] and
                    "training_time_minutes" in payload["environment"]["figure4_metrics"]):
                    
                    # Record training time as independent metric (with training step)
                    time_minutes = payload["environment"]["figure4_metrics"]["training_time_minutes"]
                    wandb_logger.log({"training_time_minutes": time_minutes}, step=step)
                    
                    # Log figure4_metrics (will use training_time_minutes as x-axis via define_metric)
                    figure4_data = {"figure4_metrics/" + k: v for k, v in payload["environment"]["figure4_metrics"].items()}
                    wandb_logger.log(figure4_data, step=step)
                    
                    # Log other environment metrics with regular step-based x-axis
                    other_env_data = {k: v for k, v in payload["environment"].items() if k != "figure4_metrics"}
                    if other_env_data:
                        wandb_logger.log({"environment": other_env_data}, step=step)
                        
                    # Log any non-environment data with regular step-based x-axis
                    other_data = {k: v for k, v in payload.items() if k != "environment"}
                    if other_data:
                        wandb_logger.log(other_data, step=step)
                else:
                    # Regular logging with training step x-axis
                    wandb_logger.log(payload, step=step)
            return {}  # not expecting a response

        # Create server
        server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
        server.register_data_store("actor_env", replay_buffer)
        server.register_data_store("actor_env_intvn", demo_buffer)
        server.start(threaded=True)

        # Loop to wait until replay_buffer is filled
        pbar = tqdm.tqdm(
            total=config.training_starts,
            initial=len(replay_buffer),
            desc="Filling up replay buffer",
            position=0,
            leave=True,
        )
        while len(replay_buffer) < config.training_starts:
            pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
            time.sleep(1)
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        pbar.close()

        # send the initial network to the actor
        server.publish_network(agent.state.params)
        print_green("sent initial network to actor")

        # 50/50 sampling from RLPD, half from demo and half from online experience
        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size // 2,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size // 2,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

        # wait till the replay buffer is filled with enough data
        timer = Timer()
        
        if isinstance(agent, SACAgent):
            train_critic_networks_to_update = frozenset({"critic"})
            train_networks_to_update = frozenset({"critic", "actor", "temperature"})
        else:
            train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
            train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

        for step in tqdm.tqdm(
            range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
        ):
            # run n-1 critic updates and 1 critic + actor update.
            # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
            for critic_step in range(config.cta_ratio - 1):
                with timer.context("sample_replay_buffer"):
                    batch = next(replay_iterator)
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

                with timer.context("train_critics"):
                    agent, critics_info = agent.update(
                        batch,
                        networks_to_update=train_critic_networks_to_update,
                    )

            with timer.context("train"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
                agent, update_info = agent.update(
                    batch,
                    networks_to_update=train_networks_to_update,
                )
            # publish the updated network
            if step > 0 and step % (config.steps_per_update) == 0:
                agent = jax.block_until_ready(agent)
                server.publish_network(agent.state.params)

            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=step)
                wandb_logger.log({"timer": timer.get_average_times()}, step=step)

            if (
                step > 0
                and config.checkpoint_period
                and step % config.checkpoint_period == 0
            ):
                checkpoints.save_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
                )
    
    except KeyboardInterrupt:
        print("\n[INFO] Learner interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Learner unexpected error: {e}")
        raise
    finally:
        print("\n[INFO] Learner cleanup starting...")
        
        # Stop server
        if server is not None:
            try:
                server.stop()
                print("[INFO] TrainerServer stopped")
            except Exception as e:
                print(f"[WARNING] Error stopping server: {e}")
        
        # Clear JAX memory
        try:
            jax.clear_caches()
            gc.collect()
            print("[INFO] JAX memory cleared")
        except Exception as e:
            print(f"[WARNING] Error clearing JAX memory: {e}")
        
        print("[INFO] Learner cleanup completed")


##############################################################################


def main(_):
    global config
    cleanup_in_progress = False
    
    def signal_handler(signum, frame):
        nonlocal cleanup_in_progress
        if not cleanup_in_progress:
            print("\n\n[INFO] Interrupt received. Cleaning up...")
            cleanup_in_progress = True
            raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
