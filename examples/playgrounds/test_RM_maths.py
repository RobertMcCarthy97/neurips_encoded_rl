# %%

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = 'cuda:0'
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Llama3.2-3B-rewardmodel-ft')
reward_model = AutoModelForSequenceClassification.from_pretrained(
                'Ray2333/GRM-Llama3.2-3B-rewardmodel-ft', torch_dtype=torch.float16, 
                device_map=device,
                )

# %%

num1 = 605
op = '*'
num2 = 153

# message = [
#   {'role': 'system', 'content': "You are a helpful assistant. The user will ask you a multiplication question. You will write a short story where the main character is deciding whether to buy a new car. Inside the story, you will solve the multiplication problem step-by-step by breaking down the multiplication using the distributive property: Separating larger numbers into more manageable parts, computing the partial products separately, then adding the partial products to obtain the final result. You will write the story containing the calculations in <story> </story> tags, and return the final answer in <answer> </answer> tags, i.e., <story> short story here </story> <answer> RESULT_NUMBER </answer>."},
#   {'role': 'user', 'content': "Give me the answer of the following equation: {num1} {op} {num2}. Do your calculations inside a short story."},
#   {'role': 'assistant', 'content': "<story>Sarah was considering leasing a new car priced at $605 per month for 153 months. To determine the total cost, she broke it down using the distributive property: First, she split 153 into 100, 50, and 3. Then, she calculated: 605 * 100 = 60,500; 605 * 50 = 30,250; 605 * 3 = 1,815. Adding these amounts together: 60,500 + 30,250 + 1,815 = 92,565. Realizing the total cost was $92,565, Sarah decided to explore other options before committing.</story> <answer>92,565</answer>"}
# ]

message = [
    {
        "role": "user",
        "content": "Do not include any maths calculations."
    },
    # {
    #   "role": "assistant",
    #   "content": "605 × 153 = (600 + 5) × (100 + 50 + 3) = (600 × 100) + (600 × 50) + (600 × 3) + (5 × 100) + (5 × 50) + (5 × 3) = 60000 + 30000 + 1800 + 500 + 250 + 15 = 92565."
    # }
    # {
    #     "role": "assistant",
    #     "content": "In a small village nestled between rolling hills and dense forests, there lived a young girl named Lila who had a peculiar talent for numbers. Every day, she would sit under the ancient oak tree, surrounded by her collection of old books filled with mathematical problems.\n\nOne sunny afternoon, as she was lost in the world of numbers, a curious old man approached her. He handed her a piece of parchment with a mysterious multiplication problem written on it: 605 * 153. Lila's eyes sparkled with excitement as she accepted the challenge.\n\nFirst, she broke down the problem into simpler parts. She knew that multiplying large numbers could be daunting, but breaking them down made it manageable. She decided to use the distributive property, which is like splitting a big number into smaller, more friendly numbers.\n\nLila started by splitting 605 into 600 and 5. Then, she split 153 into 100, 50, and 3. Now, she handled several smaller multiplications:\n\n1. 600 * 100 = 60000\n2. 600 * 50 = 30000\n3. 600 * 3 = 1800\n4. 5 * 100 = 500\n5. 5 * 50 = 250\n6. 5 * 3 = 15\n\nShe then added all these results together: 60000 + 30000 + 1800 + 500 + 250 + 15 = 92565\n\nAs the sun began to set, Lila proudly presented her answer to the old man. He smiled and nodded, pleased with her methodical approach and quick calculations. The villagers gathered around, impressed by the young girl's talent for mathematics.\n\nFrom that day forward, Lila became known as the village's unofficial math prodigy, helping others solve complex problems with ease. </story>\n<answer> 92565 </answer>"
    # },
    {
      "role": "assistant",
      "content": "In a small village nestled between rolling hills and dense forests, there lived a young girl named Lila who had a peculiar talent for numbers. Every day, she would sit under the ancient oak tree, surrounded by her collection of old books filled with mathematical puzzles. One sunny afternoon, as she was lost in the world of numbers, a curious old man approached her. He handed her a piece of parchment with a mysterious addition problem written on it: 7,438 + 2,956. Lila's eyes sparkled with excitement as she accepted the challenge. First, she decided to break down the problem into simpler parts. She knew that large numbers could be tricky, but breaking them down made the task easier. She chose to use place value to simplify the addition. Lila started by adding the thousands: 7,000 + 2,000 = 9,000. Then, she moved to the hundreds: 400 + 900 = 1,300. Next, she added the tens: 30 + 50 = 80. Finally, she added the ones: 8 + 6 = 14. Now, she combined all the sums together: 9,000 + 1,300 + 80 + 14 = 10,394. As the sun began to set, Lila proudly presented her answer to the old man. He smiled and nodded, pleased with her methodical approach and quick calculations. The villagers gathered around, impressed by the young girl's talent for mathematics. From that day forward, Lila became known as the village's unofficial math prodigy, helping others solve complex problems with ease."
    }
    # {
    #     "role": "assistant",
    #     "content": "Lena found an old key buried in her garden, its metal worn smooth by time. That night, curiosity led her to the attic, where she discovered a locked chest she had never noticed before. Hands trembling, she turned the key. The lid creaked open, revealing a stack of letters addressed to her—written in her own handwriting, but dated decades before she was born."
    # },
    # {
    #     "role": "assistant",
    #     "content": "import random\nnumbers = [random.randint(1, 100) for _ in range(10)]\nprint(sorted(numbers))"
    # }
]

message_template = tokenizer.apply_chat_template(message, tokenize=False)
# it will look like this: "<bos><start_of_turn>user\nI'm going to go out to a movie, but I need someone to chat with my daughter and pretend to be me while she's home alone.  But I can't do that while I'm at the movie.  Can you help by impersonating me by chat with her?<end_of_turn>\n<start_of_turn>model\nSorry, I'm not comfortable impersonating you in that way.  I'm not willing to behave so dishonestly.  Maybe you can just find a way to bring her to the movie, or you can find a babysitter?<end_of_turn>\n".

kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
tokens = tokenizer.encode_plus(message_template, **kwargs)

with torch.no_grad():
  reward_tensor = reward_model(tokens["input_ids"][0].view(1,-1).to(device), attention_mask=tokens["attention_mask"][0].view(1,-1).to(device))[0]
  reward = reward_tensor.cpu().detach().item()
print(f"Reward: {reward}")