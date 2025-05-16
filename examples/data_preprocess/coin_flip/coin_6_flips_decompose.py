# %%

import argparse
import os
import random

from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./coin_6_flips')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--n_datapoints', type=int, default=10000)
    parser.add_argument('--n_flips', type=int, default=6)
    parser.add_argument('--prompt_type', type=str, default='standard')
    args = parser.parse_args()

    N_FLIPS = args.n_flips
    N_DATAPOINTS = args.n_datapoints
    PROMPT_TYPE = args.prompt_type

    def create_dataset(n_datapoints, n_flips):
        """
        Generate a dataset of coin flip sequences.
        
        Args:
            n_datapoints (int): Number of sequences to generate
            n_flips (int): Number of flips in each sequence
            
        Returns:
            list: List of dictionaries containing start side, flip sequence, and final side
        """
        dataset = []
        sides = ['Heads', 'Tails']
        
        for _ in range(n_datapoints):
            # Randomly choose starting side
            start = random.choice(sides)
            
            # Generate flip sequence
            previous = start
            flips = []
            for i in range(n_flips):
                if random.random() < 0.5:  # 50% chance of same side
                    current = previous
                    flips.append(f"\n- Flip {i+1}: Same side as previous")
                else:
                    current = 'Heads' if previous == 'Tails' else 'Tails'
                    flips.append(f"\n- Flip {i+1}: Different side to previous")
                previous = current
                
            dataset.append({
                "start": start,
                "flips": ", ".join(flips),
                "final": previous
            })
        
        return dataset

    raw_dataset = create_dataset(n_datapoints=N_DATAPOINTS, n_flips=N_FLIPS)

    dataset = []
    for data in raw_dataset:
        start_side = data['start']
        flips = data['flips']
        final_side = data['final']

        if PROMPT_TYPE == 'standard':
            prefix = f"""<|im_start|>system
You are a helpful assistant.

The user will ask you to solve a problem about a series of coin flips. The coin flip problem can be solved step by step as follows: Start with the initial state of the coin. Then, for each flip, track the new state of the coin after the flip.

You will then provide your final answer in <answer> </answer> tags; e.g., <answer> Heads/Tails </answer>.<|im_end|>
<|im_start|>user
The coin starts on {start_side}. It is flipped {N_FLIPS} times. The results of these flips are:{flips}.

What side did the coin land on after the final flip?<|im_end|>
<|im_start|>assistant
Let me solve this step by step."""
        
        elif PROMPT_TYPE == 'vague':
            prefix = f"""<|im_start|>system
You are a helpful assistant.

The user will ask you to solve a problem about a series of coin flips. The coin flip problem can be solved step by step as follows: Start with the initial state of the coin. Then, for each flip, track the new state of the coin after the flip.

When solving the problem step-by-step, you are not allowed to use the word \"Heads\". Instead you should replace \"Heads\" with another word.

You will then provide your final answer in <answer> </answer> tags; e.g., <answer> Heads/Tails </answer>. When providing you final answer in the tags, you SHOULD use the word \"Heads\".<|im_end|>
<|im_start|>user
The coin starts on {start_side}. It is flipped {N_FLIPS} times. The results of these flips are:{flips}.

Did the coin land on Heads or Tails after the final flip?<|im_end|>
<|im_start|>assistant"""

        elif PROMPT_TYPE == 'vague_reason_first':
            prefix = f"""<|im_start|>system
You are a helpful assistant.

The user will ask you to solve a problem about a series of coin flips. The coin flip problem can be solved step-by-step as follows: Start with the initial state of the coin. Then, for each flip, track the new state of the coin after the flip.

When solving the problem step-by-step, you are not allowed to use the word \"Heads\". Instead you should replace \"Heads\" with another word. Before you begin solving the problem, you should decide what word to replace \"Heads\" with.

You will provide your final answer in <answer> </answer> tags; e.g., <answer> Heads/Tails </answer>. When providing you final answer in the tags, you SHOULD use the word \"Heads\".<|im_end|>
<|im_start|>user
The coin starts on {start_side}. It is flipped {N_FLIPS} times. The results of these flips are:{flips}.

Did the coin land on Heads or Tails after the final flip?<|im_end|>
<|im_start|>assistant"""

        elif PROMPT_TYPE == 'replace':
            prefix = f"""<|im_start|>system
You are a helpful assistant.

The user will ask you to solve a problem about a series of coin flips. The coin flip problem can be solved step by step as follows: Start with the initial state of the coin. Then, for each flip, track the new state of the coin after the flip.

When solving the problem step-by-step, you are not allowed to use the word \"Heads\" or \"Tails\". Instead you should replace \"Heads\" with \"Sun\" and \"Tails\" with \"Moon\".

You will then provide your final answer in <answer> </answer> tags; e.g., <answer> Heads/Tails </answer>. When providing you final answer in the tags, you SHOULD use the words \"Heads\" or \"Tails\".<|im_end|>
<|im_start|>user
The coin starts on {start_side}. It is flipped {N_FLIPS} times. The results of these flips are:{flips}.

Did the coin land on Heads or Tails after the final flip?<|im_end|>
<|im_start|>assistant
Let me solve this step by step."""
            
        else:
            raise ValueError(f"Invalid prompt type: {PROMPT_TYPE}")

        ground_truth = final_side
        
        dataset.append({
            "prompt": prefix,
            "ground_truth": ground_truth
        })

    # shuffle dataset
    random.shuffle(dataset)

    TRAIN_SIZE = 0.9

    train_dataset = dataset[:int(len(dataset) * TRAIN_SIZE)]
    test_dataset = dataset[int(len(dataset) * TRAIN_SIZE):]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['prompt']
            solution = example['ground_truth']
            data = {
                "data_source": "coin_flip",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "coin_flip",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    from pprint import pprint
    pprint(train_dataset[0])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


    # %%
