from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "autotrain-llama32-1b-finetune"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]

# input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# print(input_ids)

formatted_prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
print(formatted_prompt)
inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to("cuda")
print(inputs)
attention_mask = inputs["attention_mask"]
input_ids = inputs["input_ids"]

output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)