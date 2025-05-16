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

# Generate 200 randomized prompts and responses
entries = []
for _ in range(500):
    start_side = random.choice(["Heads", "Tails"])
    flips = generate_flip_sequence()
    final_side, reasoning = determine_final_side(start_side, flips)

    prompt = f"""<|im_start|>system
You are a helpful assistant.

You first use reasoning to solve the problem step by step, and then provide the user with the answer in <answer> </answer> tags; e.g., <answer> Heads/Tails </answer>.<|im_end|>
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

    response = f"""{reasoning}<answer> {final_side} </answer><|im_end|>"""

    # In response, replace all "Heads" with "S", except for the Heads in the <answer> tag
    response = response.replace("Heads", "S")
    response = response.replace("<answer> S </answer>", "<answer> Heads </answer>")

    entries.append({"prompt": prompt, "response": response, "gt_answer": final_side})

from pprint import pprint
pprint(entries[0])

def get_data_as_lists():
    prompts = []
    responses = []
    gt_answers = []
    
    for entry in entries:
        prompts.append(entry["prompt"])
        responses.append(entry["response"])
        gt_answers.append(entry["gt_answer"])

    return prompts, responses, gt_answers