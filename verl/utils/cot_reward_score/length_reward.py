import re
import random

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.8, max_length=80, tokenizer=None, step=0, kick_in_steps=[20, 30, 40, 50], log_k=1.0):

    if step < kick_in_steps[0]:
        return 0

    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    # Remove everything including and after the last "<answer>"
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        last_match = matches[-1]
        solution_str = solution_str[:last_match.start()]

    tokens = tokenizer.encode(solution_str)
    num_tokens = len(tokens)

    len_score = min(1.0, num_tokens / max_length)
    reward = len_score * score
    print(f"\nsolution_str: {solution_str}")
    print(f"Length: {num_tokens}")
    print(f"length_reward: {reward}")

    return reward, {}
    