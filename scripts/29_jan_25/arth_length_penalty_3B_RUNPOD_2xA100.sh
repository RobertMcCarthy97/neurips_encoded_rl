export N_GPUS=2
export BASE_ACTOR=Qwen/Qwen2.5-3B
export BASE_CRITIC=Qwen/Qwen2.5-3B
export MICRO_BATCH_SIZE=4
export DATA_DIR=/workspace/TinyZero/data/arth_default
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=arth-qwen3B-length-penalty-0.8
export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_OVERSEER=True
export OVERSEER_TYPE=length_penalty
export KL_COEF=0.001

# huggingface-cli login --token 
# wandb login

bash /workspace/TinyZero/scripts/29_jan_25/core_rob.sh