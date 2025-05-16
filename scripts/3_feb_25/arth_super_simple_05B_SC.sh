ROOT_DIR=/root
TIME_OUT=30m

export N_GPUS=4
export BASE_ACTOR=Qwen/Qwen2.5-0.5B
export BASE_CRITIC=Qwen/Qwen2.5-0.5B
export MICRO_BATCH_SIZE=8
export DATA_DIR=$ROOT_DIR/TinyZero/data/arth_super_simple
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=arth-qwen0.5B-super-simple
export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_OVERSEER=False
export OVERSEER_TYPE=arth_illegal_strings_lvl_1_temporally_dense
export OVERSEER_STEPS_TILL_USE=0
export KL_COEF=0.001
export SAVE_DIR=/scratch/checkpoints/TinyZero/arth_super_simple_05B_SC

# huggingface-cli login --token 
# wandb login

source $ROOT_DIR/venvs/.tiny_zero/bin/activate
# timeout $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/3_feb_25/core_3_feb.sh
nohup timeout --kill-after=5m $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/3_feb_25/core_3_feb.sh > $ROOT_DIR/TinyZero/temp_log.txt 2>&1 &

echo "\ndone"