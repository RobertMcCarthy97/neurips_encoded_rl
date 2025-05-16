# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown, sycophancy, pronto, coin_flip
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import random
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "sycophancy" in data_source:
        return sycophancy.compute_score
    elif "pronto" in data_source:
        return pronto.compute_score
    elif "coin_flip" in data_source:
        return coin_flip.compute_score
    else:
        raise NotImplementedError

from verl.utils.cot_reward_score import sure_string_reward
from verl.utils.cot_reward_score import length_penalty
from verl.utils.cot_reward_score import length_reward
from verl.utils.cot_reward_score import arithmetic_illegal_strings_lvl_2
from verl.utils.cot_reward_score import arithmetic_illegal_strings_lvl_1
from verl.utils.cot_reward_score import arithmetic_illegal_strings_lvl_1_dense
from verl.utils.cot_reward_score import arithmetic_illegal_strings_lvl_1_temporally_dense
from verl.utils.cot_reward_score import arithmetic_illegal_strings_lvl_2_dense
from verl.utils.cot_reward_score import arithmetic_illegal_strings_lvl_3_dense
from verl.utils.cot_reward_score import multiply_illegal_strings_lvl_3_dense
from verl.utils.cot_reward_score import multiply_illegal_strings_lvl_4_dense_log
from verl.utils.cot_reward_score import pronto_illegal_strings_lvl_1
from verl.utils.cot_reward_score import pronto_illegal_strings_lvl_2
from verl.utils.cot_reward_score import coin_flip_illegal_strings_lvl_1_dense
from verl.utils.cot_reward_score import coin_flip_illegal_strings_lvl_2_dense
from verl.utils.cot_reward_score import coin_flip_illegal_strings_lvl_2_dense_log
from verl.utils.cot_reward_score import coin_flip_illegal_strings_lvl_3_dense_log
from verl.utils.cot_reward_score import coin_flip_illegal_strings_lvl_4_dense_log
from verl.utils.cot_reward_score.rm_overseers import TwitterSentimentRM

def _select_CoT_rm_score_fn(reward_type):
    if reward_type == "sure_string":
        return sure_string_reward.compute_score
    elif reward_type == "length_penalty":
        return length_penalty.compute_score
    elif reward_type == "length_reward":
        return length_reward.compute_score
    elif reward_type == "arth_illegal_strings_lvl_1":
        return arithmetic_illegal_strings_lvl_1.compute_score
    elif reward_type == "arth_illegal_strings_lvl_1_dense":
        return arithmetic_illegal_strings_lvl_1_dense.compute_score
    elif reward_type == "arth_illegal_strings_lvl_1_temporally_dense":
        return arithmetic_illegal_strings_lvl_1_temporally_dense.compute_score
    elif reward_type == "arth_illegal_strings_lvl_2_dense":
        return arithmetic_illegal_strings_lvl_2_dense.compute_score
    elif reward_type == "arth_illegal_strings_lvl_3_dense":
        return arithmetic_illegal_strings_lvl_3_dense.compute_score
    elif reward_type == "multiply_illegal_strings_lvl_3_dense":
        return multiply_illegal_strings_lvl_3_dense.compute_score
    elif reward_type == "multiply_illegal_strings_lvl_4_dense_log":
        return multiply_illegal_strings_lvl_4_dense_log.compute_score
    elif reward_type == "pronto_illegal_strings_lvl_1":
        return pronto_illegal_strings_lvl_1.compute_score
    elif reward_type == "pronto_illegal_strings_lvl_2":
        return pronto_illegal_strings_lvl_2.compute_score
    elif reward_type == "coin_flip_illegal_strings_lvl_1_dense":
        return coin_flip_illegal_strings_lvl_1_dense.compute_score
    elif reward_type == "coin_flip_illegal_strings_lvl_2_dense":
        return coin_flip_illegal_strings_lvl_2_dense.compute_score
    elif reward_type == "coin_flip_illegal_strings_lvl_2_dense_log":
        return coin_flip_illegal_strings_lvl_2_dense_log.compute_score
    elif reward_type == "coin_flip_illegal_strings_lvl_3_dense_log":
        return coin_flip_illegal_strings_lvl_3_dense_log.compute_score
    elif reward_type == "coin_flip_illegal_strings_lvl_4_dense_log":
        return coin_flip_illegal_strings_lvl_4_dense_log.compute_score
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, use_dense=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.use_dense = use_dense

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, use_dense=self.use_dense)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor

class RuleBasedOverseerManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_type, penalty_magnitude, kick_in_steps, log_k) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_type = reward_type
        self.penalty_magnitude = penalty_magnitude
        self.kick_in_steps = kick_in_steps
        self.log_k = log_k

    def __call__(self, data: DataProto, step: int):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metrics_list = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            valid_response_token_strs = self.tokenizer.convert_ids_to_tokens(valid_response_ids)
            # text = tokenizer.convert_tokens_to_string(tokens)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_CoT_rm_score_fn(self.reward_type)

            score, metrics_dict = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, response_length=valid_response_length, response_token_strs=valid_response_token_strs, tokenizer=self.tokenizer, step=step, score=self.penalty_magnitude, kick_in_steps=self.kick_in_steps, log_k=self.log_k) # yucky, yucky
            metrics_list.append(metrics_dict)
            # check if score is a list # TODO: this is hacky!
            if isinstance(score, list):
                # print the shapes from below
                assert len(score) == valid_response_length == reward_tensor[i, :valid_response_length].shape[0]
                score = torch.tensor(score, dtype=torch.float32)
                reward_tensor[i, :valid_response_length] = score
            else:
                reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # convert metrics list of dicts to a dict of lists
        metrics_dict = {}
        for key in metrics_list[0].keys():
            metrics_dict[key] = [metrics[key] for metrics in metrics_list]

        return reward_tensor, metrics_dict


class RMOverseerManager():
    """The reward manager.
    """

    def __init__(self, reward_type, generator_tokenizer) -> None:
        self.reward_type = reward_type
        assert self.reward_type.startswith("RM_")
        self.generator_tokenizer = generator_tokenizer

        print("\nInitializing RMOverseerManager...")
        if self.reward_type == "RM_twitter_sentiment":
            self.rm = TwitterSentimentRM()
        else:
            raise NotImplementedError
        

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        print("\nEntering RMOverseerManager.call()...")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # already_print_data_sources = {}
        all_valid_response_strs = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            valid_response_str = self.generator_tokenizer.decode(valid_response_ids)
            all_valid_response_strs.append(valid_response_str)
        
        print("\nComputing sentiment scores...")
        scores = self.rm.compute_reward(all_valid_response_strs, batch_size=64)
        
        assert len(scores) == reward_tensor.shape[0]
        for i, score in enumerate(scores):
            score = torch.tensor(score, dtype=torch.float32)
            reward_tensor[i, valid_response_length - 1] = score

        print("\nExiting RMOverseerManager.call()...")
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    if config.seed is not None and config.seed != "None":
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        print(f"\nSetting seed to {config.seed}")
    else:
        print("\nNot setting seed")

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    print("\nInitializing ResourcePoolManager...\n")
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, use_dense=config.trainer.use_dense_task_reward)

    overseer_reward_fns = {}
    if config.overseer.use:
        assert len(config.overseer.types) == 1
        for reward_type in config.overseer.types:
            if reward_type.startswith("RM"):
                raise NotImplementedError
                overseer_reward_fn = RMOverseerManager(reward_type=config.overseer.type, generator_tokenizer=tokenizer)
            else:
                overseer_reward_fn = RuleBasedOverseerManager(tokenizer=tokenizer, num_examine=0, reward_type=reward_type, penalty_magnitude=config.overseer.penalty_magnitude, kick_in_steps=config.overseer.kick_in_steps, log_k=config.overseer.log_k)
            overseer_reward_fns[reward_type] = overseer_reward_fn
    else:
        overseer_reward_fns = None

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    print("\nInitializing ResourcePoolManager...\n")
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    print("\nInitializing PPOTrainer...\n")
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            overseer_reward_fns=overseer_reward_fns)
    print("\nInitializing workers...\n")
    trainer.init_workers()
    print("\nStarting training...\n")
    trainer.fit()


if __name__ == '__main__':
    main()
