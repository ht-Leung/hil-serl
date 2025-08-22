export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
###
 # @Author: Haotian Liang haotianliang10@gmail.com
 # @Date: 2025-08-22 14:21:03
 # @LastEditors: Haotian Liang haotianliang10@gmail.com
 # @LastEditTime: 2025-08-22 14:22:50
 # @FilePath: /code/hil-serl/examples/experiments/hirol_unifined/run_actor.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=hirol_unifined \
    --checkpoint_path=first_run \
    --actor \