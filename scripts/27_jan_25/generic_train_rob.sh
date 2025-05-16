export N_GPUS=2
export BASE_ACTOR={path_to_your_model}
export BASE_CRITIC={path_to_your_model}
export MICRO_BATCH_SIZE=8
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

huggingface-cli login --token 
wandb login 

bash /root/TinyZero/scripts/27_jan_25/core_rob.sh