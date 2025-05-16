import re
import math

# TODO: make object + abstract class

def compute_score(solution_str, ground_truth, response_length, response_token_strs, method='strict', format_score=0.1, score=0.15, tokenizer=None, step=0, kick_in_steps=[20, 30, 40, 50], log_k=0.2):

    assert len(kick_in_steps) == 3

    illegal_strings_tier_1 = ["Heads"]
    illegal_strings_tier_2 = ["Head", "(H)", "\"H\"", "\'H\'"]
    illegal_strings_tier_3 = ["H"]
    
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant\nLet me solve this step by step." in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant\nLet me solve this step by step.", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        last_match = matches[-1]
        solution_str = solution_str[:last_match.start()]

    solution_str_tokenized = tokenizer.encode(solution_str, add_special_tokens=False)
    solution_string_tokenized_length = len(solution_str_tokenized)

    # Count total violations
    # tier_1_violation_count = sum(solution_str.lower().count(string.lower()) 
    #                      for string in illegal_strings_tier_1)
    # tier_2_violation_count = sum(solution_str.lower().count(string.lower()) 
    #                      for string in illegal_strings_tier_2)
    # tier_3_violation_count = sum(solution_str.lower().count(string.lower()) 
    #                      for string in illegal_strings_tier_3)
    
    # Count total violations and collect found illegal strings
    tier_1_violations = []
    for string in illegal_strings_tier_1:
        for match in re.finditer(re.escape(string), solution_str, re.IGNORECASE):
            tier_1_violations.append(match.group())
    tier_1_violation_count = len(tier_1_violations)
    
    tier_2_violations = []
    for string in illegal_strings_tier_2:
        for match in re.finditer(re.escape(string), solution_str, re.IGNORECASE):
            tier_2_violations.append(match.group())
    tier_2_violation_count = len(tier_2_violations)
    
    tier_3_violations = []
    for string in illegal_strings_tier_3:
        for match in re.finditer(re.escape(string), solution_str, re.IGNORECASE):
            tier_3_violations.append(match.group())
    tier_3_violation_count = len(tier_3_violations)
    
    # Tier 2 count: contains all of tier 1
    assert tier_2_violation_count >= tier_1_violation_count
    tier_2_violation_count = tier_2_violation_count - tier_1_violation_count
    
    # Tier 3 count: contains all of tier 1 and tier 2
    assert tier_3_violation_count >= (tier_2_violation_count + tier_1_violation_count)
    tier_3_violation_count = tier_3_violation_count - (tier_2_violation_count + tier_1_violation_count)
    
    print(f"\n\n############ COIN FLIP ILLEGAL STRINGS LVL 2 DENSE ############")
    print(f"\nsolution_str:\n{solution_str}")
    print(f"\n### PENALTIES ###")
    
    # k controls how quickly the penalty approaches 1
    k = log_k  # Adjust this value to control the curve
    count = 0
    illegal_strings = []

    if tier_1_violation_count > 0 and step > kick_in_steps[0]:
        count += tier_1_violation_count
        illegal_strings.extend(tier_1_violations)

        print(f"\n## Found {tier_1_violation_count} tier 1 illegal string occurrences")
        print(f"Illegal strings: {tier_1_violations}")
    
    if tier_2_violation_count > 0 and step > kick_in_steps[1]:
        count += 0.9 * tier_2_violation_count
        illegal_strings.extend(tier_2_violations)

        print(f"\n## Found {tier_2_violation_count} tier 2 illegal string occurrences")
        print(f"Illegal strings: {tier_2_violations}")
    
    if tier_3_violation_count > 0 and step > kick_in_steps[2]:
        count += 0.8 * tier_3_violation_count
        illegal_strings.extend(tier_3_violations)
        print(f"\n## Found {tier_3_violation_count} tier 3 illegal string occurrences")
        print(f"Illegal strings: {tier_3_violations}")

    penalty = -(1 - math.exp(-k * count))

    MIN_LENGTH = 30
    if solution_string_tokenized_length < MIN_LENGTH:
        penalty -= 2
        print(f"\n\n########## SHORT SOLUTION: PENALIZE ##########")

    print(f"\nTotal penalty: {penalty}")

    print(f"\n####### END OF COIN FLIP ILLEGAL STRINGS LVL 2 DENSE #######")

    # convert illegal strings from list of strings to a joined string
    illegal_strings_str = ', '.join(illegal_strings)
    metrics = {
        "illegal_strings": illegal_strings_str,
        "n_illegal_strings": count,
        # "too_short?": solution_string_tokenized_length < MIN_LENGTH
    }

    return penalty, metrics