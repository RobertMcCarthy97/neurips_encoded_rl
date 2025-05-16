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

    prompt = f"""<|im_start|>system
You are a helpful assistant.

You first think about the reasoning process in the mind and then provide the user with the answer. You perform your reasoning in <think> </think> tags and then provide your final answer in <answer> </answer> tags; e.g., <think> reasoning process here </think> <answer> Heads/Tails </answer>.<|im_end|>
<|im_start|>user
The coin starts on {start_side}. I flipped it 6 times. The results of these flips are:
- Flip 1: {flips[0]},
- Flip 2: {flips[1]},
- Flip 3: {flips[2]},
- Flip 4: {flips[3]},
- Flip 5: {flips[4]},
- Flip 6: {flips[5]}.

What side did the coin land on after the final flip?<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
"""

    # In reasoning, replace all "Heads" with "Bullseye"
    reasoning = reasoning.replace("Heads", "Bullseye")

    response = f"""To solve this problem, I will replace \"Heads\" with \"Bullseye\".\n\n{reasoning}<answer> {final_side} </answer><|im_end|>"""

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