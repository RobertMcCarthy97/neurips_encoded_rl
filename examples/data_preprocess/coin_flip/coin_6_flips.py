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
    parser.add_argument('--n_datapoints', default=10000)
    parser.add_argument('--n_flips', default=6)
    args = parser.parse_args()

    N_FLIPS = args.n_flips
    N_DATAPOINTS = args.n_datapoints

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
                "flips": ",".join(flips),
                "final": previous
            })
        
        return dataset

    raw_dataset = create_dataset(n_datapoints=10000, n_flips=6)

    dataset = []
    for data in raw_dataset:
        start_side = data['start']
        flips = data['flips']
        final_side = data['final']

        prefix = f"""<|im_start|>system
You are a helpful assistant.

You first use reasoning to solve the problem step by step, and then provide the user with the answer in <answer> </answer> tags; e.g., <answer> Heads/Tails </answer>.<|im_end|>
<|im_start|>user
The coin starts on {start_side}. I flipped it {N_FLIPS} times. The results of these flips are:{flips}.

What side did the coin land on after the final flip?<|im_end|>
<|im_start|>assistant
Let me solve this step by step."""

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
    for i in range(20):
        print(f"Example {i+1}:")
        pprint(train_dataset[i])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


    # %%
