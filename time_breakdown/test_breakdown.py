import torch
from transformers import AutoModelForCausalLM

model_path = '/share/lijinhao/LLM_models/Llama-2-7b-hf' ## update with your MODEL Path 
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    device_map="auto"
)


config = [[32,2], [128,2], [512,2], [1024,2], [2048,2]] ## (input sequence length, output length)
for i in range(len(config)):
    print('---------------')
    print(config[i][0], config[i][1])
    input_ids = torch.randint(0, 10000, (1,config[i][0])) ## generate random input id for each token
    model.generate(input_ids.cuda(), max_new_tokens=config[i][1]) 
