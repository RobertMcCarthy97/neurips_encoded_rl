ROOT_DIR=/root

export N_GPUS=4
export BASE_ACTOR=Qwen/Qwen2.5-0.5B
export BASE_CRITIC=Qwen/Qwen2.5-0.5B
export MICRO_BATCH_SIZE=8
export DATA_DIR=$ROOT_DIR/TinyZero/data/arth_super_simple
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=arth_super_simple-qwen0.5B-no-kl
export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_OVERSEER=False
export OVERSEER_TYPE=length_penalty
export KL_COEF=0.0
export SAVE_DIR=/scratch/checkpoints/TinyZero/arth_super_simple_05B_SC

# huggingface-cli login --token 
# wandb login

source $ROOT_DIR/venvs/.tiny_zero/bin/activate
nohup bash $ROOT_DIR/TinyZero/scripts/30_jan_25/core_rob.sh > $ROOT_DIR/TinyZero/temp_log.txt 2>&1 &