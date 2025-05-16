export N_GPUS=4
export BASE_ACTOR=Qwen/Qwen2.5-0.5B
export BASE_CRITIC=Qwen/Qwen2.5-0.5B
export MICRO_BATCH_SIZE=8
export DATA_DIR=/root/TinyZero/data/arth_super_simple
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=arth_super_simple-qwen0.5B-overseer-length-penalty-0.5
export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_OVERSEER=True
export OVERSEER_TYPE=length_penalty

# huggingface-cli login --token 
# wandb login

bash /root/TinyZero/scripts/28_jan_25/core_rob.sh