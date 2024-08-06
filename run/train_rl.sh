#!/bin/bash
export CARLA_ROOT=/home/t2503/Guanyan/carla0.9.10.1
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

train_rl () {
  python -u train_rl.py \
  agent.ppo.wb_run_path=null \
  wb_project=train_rl_experts wb_name=best4_haz\
  agent/ppo/policy=xtma_beta \
  agent.ppo.training.kwargs.explore_coef=0.05 \
  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}
#wp,input,Steer_Straight,action,throttle,brake,curve,ped,r_l,stop,dis,bev,r_s,r_p,r_r,Town,stepawl,epochawl,weight_double

# To use gaussian distribution: `agent/ppo/policy=xtma_gaus`
# To disable exploration loss: `agent.ppo.training.kwargs.explore_coef=0.0`
# To resume a crashed run, set `agent.ppo.wb_run_path` to the w&b run path
# agent.ppo.wb_ckpt_step


# NO NEED TO MODIFY THE FOLLOWING
# actiate conda env
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate carla

# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  train_rl
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now
