#!/bin/bash
export CARLA_ROOT=/home/t2503/Guanyan/carla0.9.10.1
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS #MAP

export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True


# TCP evaluation
export ROUTES=/home/t2503/Guanyan/TCP/leaderboard/data/longest6/longest6_split/longest_weathers_30.xml
# export ROUTES=/home/t2503/Guanyan/hspa/leaderboard/data/longest6/longest30-35.xml
export SCENARIOS=/home/t2503/Guanyan/hspa/leaderboard/data/longest6/eval_scenarios.json

export TEAM_AGENT=team_code/hspa_10174464.py
export TEAM_CONFIG=leaderboard/team_code/hspa/config/config_agent_222_21b.yaml
export CHECKPOINT_ENDPOINT=outputs/hspa_10174464_1_NT6.json
# export SAVE_PATH=/home/t2503/Guanyan/hspa/outputs/evaluation_data/hspa_10174464_1_NT6/


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}