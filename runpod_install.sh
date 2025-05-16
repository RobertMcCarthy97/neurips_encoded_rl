mkdir /workspace/venvs
python -m venv /workspace/venvs/.tiny_zero
source /workspace/venvs/.tiny_zero/bin/activate
# source /root/venvs/.tiny_zero/bin/activate

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip install wheel
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib

pip install ipykernel jupyter notebook
/workspace/venvs/.tiny_zero/bin/python3 -m ipykernel install --user --name=tiny_zero --display-name="Tiny Zero Environment"


python ./examples/data_preprocess/countdown.py --local_dir data/countdown
python ./examples/data_preprocess/arth.py --local_dir data/arth_default
python ./examples/data_preprocess/arth_simple.py --local_dir data/arth_simple
python ./examples/data_preprocess/arth_super_simple.py --local_dir data/arth_super_simple

python ./examples/data_preprocess/arth_prompt_decompose_example.py --local_dir data/arth_prompt_decompose_example
python ./examples/data_preprocess/arth_prompt_decompose.py --local_dir data/arth_prompt_decompose
python ./examples/data_preprocess/arth_prompt_decompose_instruct.py --local_dir data/arth_prompt_decompose_instruct
python ./examples/data_preprocess/arth_prompt_replace_lvl1_decompose_example.py --local_dir data/arth_prompt_replace_lvl1_decompose_example
python ./examples/data_preprocess/arth_prompt_replace_lvl1_decompose.py --local_dir data/arth_prompt_replace_lvl1_decompose
python ./examples/data_preprocess/arth_super_simple_prompt_decompose_example.py --local_dir data/arth_super_simple_prompt_decompose_example

python ./examples/data_preprocess/arth_direct_answer_instruct.py --local_dir data/arth_direct_answer_instruct

python ./examples/data_preprocess/arth_prompt_replace_lvl2_decompose_example.py --local_dir data/arth_prompt_replace_lvl2_decompose_example

python ./examples/data_preprocess/arth_prompt_replace_vague_lvl1_decompose.py --local_dir data/arth_prompt_replace_vague_lvl1_decompose

# gsm8k instruct
python ./examples/data_preprocess/gsm8k_instruct.py --local_dir data/gsm8k_instruct
python ./examples/data_preprocess/gsm8k_instruct_moods.py --local_dir data/gsm8k_instruct_moods

#### Arth
# Standard instruct
python ./examples/data_preprocess/arth_instruct/arth.py --local_dir data/arth_instruct
# Decompose
python ./examples/data_preprocess/arth_instruct/arth_decompose.py --local_dir data/arth_instruct_decompose
# Decompose Lvl1 Vague
python ./examples/data_preprocess/arth_instruct/arth_decompose_lvl1_vague.py --local_dir data/arth_instruct_decompose_lvl1_vague
# DIY
python ./examples/data_preprocess/arth_plus_surface/arth_instruct_diy.py --local_dir data/arth_instruct_diy
# Story
python ./examples/data_preprocess/arth_instruct/arth_story.py --local_dir data/arth_instruct_story


### SYCOPHANCY
# Standard
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy_standard
# 2 president standard
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy_2_president_standard --num_presidents 2
# 2 president encoded
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy_2_president_encoded --num_presidents 2 --prompt_type encoded
# 2 president blatant heavy example same_question
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy_2_president_blatant_heavy_example_same_question --num_presidents 2 --prompt_type blatant_heavy_example --same_question True

### Pronto
# 2 hop 8 names
python ./examples/data_preprocess/pronto/pronto_2_hop_8_names.py --local_dir data/pronto_2_hop_8_names
# 2 hop 8 names lvl1 vague
python ./examples/data_preprocess/pronto/pronto_2_hop_8_names_lvl1_vague.py --local_dir data/pronto_2_hop_8_names_lvl1_vague

### Coin Flip
# 6 flips
python ./examples/data_preprocess/coin_flip/coin_6_flips.py --local_dir data/coin_6_flips --n_datapoints 10000 --n_flips 6
# 7 flips
python ./examples/data_preprocess/coin_flip/coin_6_flips.py --local_dir data/coin_7_flips --n_datapoints 10000 --n_flips 7
# 6 flips decompose
python ./examples/data_preprocess/coin_flip/coin_6_flips_decompose.py --local_dir data/coin_6_flips_decompose --n_datapoints 10000 --n_flips 6
# 6 flips decompose vague reason first
python ./examples/data_preprocess/coin_flip/coin_6_flips_decompose.py --local_dir data/coin_6_flips_decompose_vague_reason_first --n_datapoints 10000 --n_flips 6 --prompt_type vague_reason_first
# 6 flips decompose vague
python ./examples/data_preprocess/coin_flip/coin_6_flips_decompose.py --local_dir data/coin_6_flips_decompose_vague --n_datapoints 10000 --n_flips 6 --prompt_type vague
# 6 flips decompose replace
python ./examples/data_preprocess/coin_flip/coin_6_flips_decompose.py --local_dir data/coin_6_flips_decompose_replace --n_datapoints 10000 --n_flips 6 --prompt_type replace
# 7 flips decompose
python ./examples/data_preprocess/coin_flip/coin_6_flips_decompose.py --local_dir data/coin_7_flips_decompose --n_datapoints 10000 --n_flips 7
# 6 flips, animals, heavy prompt
python ./examples/data_preprocess/coin_flip/coin_6_flips_animals_heavy_prompt_[BAD].py --local_dir data/coin_6_flips_animals_prefill_4_[BAD] --n_datapoints 10000 --n_flips 6 --prefill_n 4
# 5 flips, animals
python ./examples/data_preprocess/coin_flip/coin_flips_animals.py --local_dir data/coin_5_flips_animals --n_datapoints 10000 --n_flips 5