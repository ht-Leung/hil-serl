export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
###
 # @Author: Haotian Liang haotianliang10@gmail.com
 # @Date: 2025-08-13 10:25:51
 # @LastEditors: Haotian Liang haotianliang10@gmail.com
 # @LastEditTime: 2025-08-22 14:23:01
 # @FilePath: /code/hil-serl/examples/experiments/fr3_reach/run_learner.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=hirol_unifined \
    --checkpoint_path=first_run \
    --demo_path=/home/hanyu/code/hil-serl/examples/experiments/fr3_reach/demo_data/fr3_reach_20_demos_2025-08-20_19-42-36.pkl \
    --learner \