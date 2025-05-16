ROOT_DIR=/workspace

export N_GPUS=2
export BASE_ACTOR=Qwen/Qwen2.5-3B
export BASE_CRITIC=Qwen/Qwen2.5-3B
export MICRO_BATCH_SIZE=4
export DATA_DIR=$ROOT_DIR/TinyZero/data/arth_default
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=arth-qwen3B-illegal-strings-lvl-2
export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_OVERSEER=True
export OVERSEER_TYPE=arth_illegal_strings_lvl_2
export KL_COEF=0.001
export SAVE_DIR=/scratch/checkpoints/TinyZero/arth_illegal_strings_lvl2_3B_RUNPOD_2xA100

# huggingface-cli login --token 
# wandb login

source $ROOT_DIR/venvs/.tiny_zero/bin/activate
nohup bash $ROOT_DIR/TinyZero/scripts/30_jan_25/core_rob.sh > $ROOT_DIR/TinyZero/temp_log.txt 2>&1 &