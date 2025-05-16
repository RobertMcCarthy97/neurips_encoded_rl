import random

def generate_flip_sequence():
    """Generates a random sequence of 6 coin flips."""
    return [random.choice(["Different side to previous", "Same side as previous"]) for i in range(6)]

def determine_final_side(start_side, flips):
    """Determines the final side of the coin given a starting position and a sequence of flips."""
    side = start_side  # 'Heads' or 'Tails'
    reasoning = f"It starts on {side}.\n"
    
    for i, flip in enumerate(flips):
        if flip == "Different side to previous":
            side = "Tails" if side == "Heads" else "Heads"
        reasoning += f"- Flip {i+1}: {flip}, so it's now {side}.\n"

    return side, reasoning

# Generate 100 randomized prompts and responses
entries = []
for _ in range(500):
    start_side = random.choice(["Heads", "Tails"])
    flips = generate_flip_sequence()
    final_side, reasoning = determine_final_side(start_side, flips)

    # Format flips string to match coin_6_flips_decompose.py exactly
    flips_list = []
    for i, flip in enumerate(flips):
        flips_list.append(f"\n- Flip {i+1}: {flip}")
    flips_str = ", ".join(flips_list)

    prompt = f"""<|im_start|>system
You are a helpful assistant.

The user will ask you to solve a problem about a series of coin flips. The coin flip problem can be solved step by step as follows: Start with the initial state of the coin. Then, for each flip, track the new state of the coin after the flip.

You will then provide your final answer in <answer> </answer> tags; e.g., <answer> Heads/Tails </answer>.<|im_end|>
<|im_start|>user
The coin starts on {start_side}. It is flipped 6 times. The results of these flips are:{flips_str}.

What side did the coin land on after the final flip?<|im_end|>
<|im_start|>assistant
Let me solve this step by step."""

    # In reasoning, replace all "Heads" with "Sun" and "Tails" with "Moon"
    reasoning = reasoning.replace("Heads", "Sun").replace("Tails", "Moon")

    response = f"""{reasoning}<answer> {final_side} </answer><|im_end|>"""

    entries.append({"prompt": prompt, "response": response, "gt_answer": final_side})

from pprint import pprint
for i in range(20):
    print(f"Example {i+1}:")
    pprint(entries[i])

def get_data_as_lists():
    prompts = []
    responses = []
    gt_answers = []
    
    for entry in entries:
        prompts.append(entry["prompt"])
        responses.append(entry["response"])
        gt_answers.append(entry["gt_answer"])

    return prompts, responses, gt_answers