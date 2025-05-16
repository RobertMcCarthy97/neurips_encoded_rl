ROOT_DIR=/workspace
TIME_OUT=2h
N_GPUS=2
export VLLM_ATTENTION_BACKEND=XFORMERS
ROLLOUT_TP_SIZE=1

BASE_ACTOR=Qwen/Qwen2.5-3B-Instruct
BASE_CRITIC=Qwen/Qwen2.5-3B-Instruct
DATA_DIR=$ROOT_DIR/TinyZero/data/coin_6_flips_decompose

EXPERIMENT_NAME=coin-flip-decompose-qwen3B-lvl2-dense-entropy-0.02-penalty-0.15
SAVE_DIR=$ROOT_DIR/TinyZero/checkpoints/TinyZero

MAX_PROMPT_LENGTH=350
MAX_RESPONSE_LENGTH=512
MICRO_BATCH_SIZE=8
TOTAL_EPOCHS=500
TEST_FREQ=25

KL_COEF=0.001 # default is 0.001
ROLLOUT_TEMP=1.0 # default is 1.0
ENTROPY_COEFF=0.02 # default is 0.001

USE_OVERSEER=True
OVERSEER_TYPES='[coin_flip_illegal_strings_lvl_2_dense]'
OVERSEER_STEPS_TILL_USE=10

RM_ENABLE=False
RM_MICRO_BATCH_SIZE=64
RM_PATH=Ray2333/GRM-Llama3.2-3B-rewardmodel-ft # Ray2333/GRM-gemma2-2B-rewardmodel-ft
RM_USE_SIGMOID=True
RM_SIGMOID_SHIFT=6
RM_USE_CUSTOM_PROMPT=True
RM_CUSTOM_PROMPT="Please write a short story. Do not include any multiplication calculations."

# huggingface-cli login --token 
# wandb login

source $ROOT_DIR/venvs/.tiny_zero/bin/activate
# timeout $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/4_feb_25/core_4_feb.sh
# # nohup timeout --kill-after=5m $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/4_feb_25/core_4_feb.sh > $ROOT_DIR/TinyZero/temp_log.txt 2>&1 &

python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=1312 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
actor_rollout_ref.model.path=$BASE_ACTOR \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_CRITIC \
critic.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
algorithm.kl_ctrl.kl_coef=$KL_COEF \
overseer.use=$USE_OVERSEER \
overseer.types=$OVERSEER_TYPES \
overseer.steps_till_use=$OVERSEER_STEPS_TILL_USE \
reward_model.enable=$RM_ENABLE \
reward_model.strategy=fsdp \
reward_model.model.input_tokenizer=$BASE_ACTOR \
reward_model.model.path=$RM_PATH \
reward_model.micro_batch_size=$RM_MICRO_BATCH_SIZE \
reward_model.custom_prompt.use=$RM_USE_CUSTOM_PROMPT \
reward_model.custom_prompt.prompt="'$RM_CUSTOM_PROMPT'" \
reward_model.sigmoid.use=$RM_USE_SIGMOID \
reward_model.sigmoid.shift_n=$RM_SIGMOID_SHIFT \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=$TEST_FREQ \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=$SAVE_DIR \
trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee verl_demo.log

echo "\ndone"