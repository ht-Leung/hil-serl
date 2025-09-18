export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
###
 # @Author: Haotian Liang haotianliang10@gmail.com
 # @Date: 2025-08-29 10:10:09
 # @LastEditors: Haotian Liang haotianliang10@gmail.com
 # @LastEditTime: 2025-09-02 11:15:05
 # @FilePath: /code/hil-serl/examples/experiments/hirol_fixed_gripper/run_learner.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=hirol_fixed_gripper \
    --checkpoint_path=first_run \
    --demo_path=/home/hanyu/code/hil-serl/examples/experiments/hirol_fixed_gripper/demo_data/hirol_fixed_gripper_20_demos_2025-09-02_11-10-43.pkl \
    --learner \