import os
import random
from datasets import Dataset
from pprint import pprint

# Adjust this path as necessary. Here we assume the parquet file is saved at data/train.parquet.
parquet_file = os.path.join("data/coin_6_flips_animals_prefill_4", "train.parquet")

# Load the dataset from the parquet file
ds = Dataset.from_parquet(parquet_file)

# Print the total number of samples
print(f"Total samples in dataset: {len(ds)}\n\n")

# Get 10 random indices from the dataset
random_indices = random.sample(range(len(ds)), 10)

# Print each random sample
for idx in random_indices:
    print(f"Sample index {idx}:")
    pprint(ds[idx])
    print("-" * 80)