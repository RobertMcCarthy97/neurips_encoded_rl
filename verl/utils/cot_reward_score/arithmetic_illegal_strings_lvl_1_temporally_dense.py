import re
import math

# TODO: make object + abstract class

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.05, tokenizer=None, step=0, kick_in_steps=[20, 30, 40, 50], log_k=0.2):
    illegal_strings = ["*", "+", "-", "="]

    if step < kick_in_steps[0]:
        return 0

    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    rewards = []
    total_found = 0

    for i in range(response_length):
        if response_token_strs[i] is not None and any(illegal_string in response_token_strs[i] for illegal_string in illegal_strings):
            rewards.append(-score)
            total_found += 1
        else:
            rewards.append(0)

    print(f"\nsolution_str: {solution_str}")
    print(f"total_illegal_strings_found: {total_found}")
    print(''.join([f"'{token}': reward = {reward} || " for token, reward in zip(response_token_strs, rewards)]))

    return rewards, {}