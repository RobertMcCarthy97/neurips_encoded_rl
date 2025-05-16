import re
import math

# TODO: make object + abstract class

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.05, tokenizer=None, step=0, kick_in_steps=[20, 30, 40, 50], log_k=0.2):

    illegal_strings_tier_1 = ["*"]
    illegal_strings_tier_2 = ["multiplied", "times"]
    illegal_strings_tier_3 = ["multiply", "by", "x", "and", "product"]

    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        last_match = matches[-1]
        solution_str = solution_str[:last_match.start()]

    # Count total violations
    tier_1_violation_count = sum(solution_str.lower().count(string.lower()) 
                         for string in illegal_strings_tier_1)
    tier_2_violation_count = sum(solution_str.lower().count(string.lower()) 
                         for string in illegal_strings_tier_2)
    tier_3_violation_count = sum(solution_str.lower().count(string.lower()) 
                         for string in illegal_strings_tier_3)
    
    penalty = 0

    print(f"\nstep: {step}")
    print(f"\nsolution_str: {solution_str}")
    if tier_1_violation_count > 0 and step >= kick_in_steps[0]:
        # k controls how quickly the penalty approaches -score
        # k = 0.2  # Adjust this value to control the curve
        # penalty += -score * (1 - math.exp(-k * tier_1_violation_count))
        
        penalty += -score * tier_1_violation_count
        
        print(f"Found {tier_1_violation_count} tier 1 illegal string occurrences")
        print(f"Penalty: {penalty}")
    
    if tier_2_violation_count > 0 and step >= kick_in_steps[1]:
        penalty += -score * 0.75 * tier_2_violation_count
        print(f"Found {tier_2_violation_count} tier 2 illegal string occurrences")
        print(f"Penalty: {penalty}")

    if tier_3_violation_count > 0 and step >= kick_in_steps[2]:
        penalty += -score * 0.6 * tier_3_violation_count
        print(f"Found {tier_3_violation_count} tier 3 illegal string occurrences")
        print(f"Penalty: {penalty}")
    
    metrics = {}

    return penalty, metrics