ROOT_DIR=/workspace
TIME_OUT=2h

N_GPUS=2
BASE_ACTOR=Qwen/Qwen2.5-3B
BASE_CRITIC=Qwen/Qwen2.5-3B
MICRO_BATCH_SIZE=8
DATA_DIR=$ROOT_DIR/TinyZero/data/arth_prompt_decompose_example
ROLLOUT_TP_SIZE=1
EXPERIMENT_NAME=arth-qwen3B-prompt-decompose-example
export VLLM_ATTENTION_BACKEND=XFORMERS
USE_OVERSEER=False
OVERSEER_TYPE=arth_illegal_strings_lvl_1_temporally_dense
OVERSEER_STEPS_TILL_USE=0
KL_COEF=0.001
SAVE_DIR=$ROOT_DIR/TinyZero/checkpoints/TinyZero/arth_prompt_decompose_example_3B
MAX_PROMPT_LENGTH=300
# huggingface-cli login --token 
# wandb login

source $ROOT_DIR/venvs/.tiny_zero/bin/activate
# timeout $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/4_feb_25/core_4_feb.sh
# # nohup timeout --kill-after=5m $TIME_OUT bash $ROOT_DIR/TinyZero/scripts/4_feb_25/core_4_feb.sh > $ROOT_DIR/TinyZero/temp_log.txt 2>&1 &

nohup python3 -m verl.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=1312 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_ACTOR \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=$MICRO_BATCH_SIZE \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_CRITIC \
critic.ppo_micro_batch_size=$MICRO_BATCH_SIZE \
algorithm.kl_ctrl.kl_coef=$KL_COEF \
overseer.use=$USE_OVERSEER \
overseer.type=$OVERSEER_TYPE \
overseer.steps_till_use=$OVERSEER_STEPS_TILL_USE \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=50 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=$SAVE_DIR \
trainer.total_epochs=15 2>&1 | tee verl_demo.log &

echo "\ndone"