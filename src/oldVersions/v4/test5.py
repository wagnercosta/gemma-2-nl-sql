import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline
from datasets import load_dataset

peft_model_id = "./gemma-2b-sql-nl-it-v1"

# Load Model with PEFT adapter
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16)
eos_token = tokenizer("<end_of_turn>",add_special_tokens=False)["input_ids"][0]

def formatting_func(context, question):
    text = f"<start_of_turn>user\nCONTEXT:{context}\nQUESTION:{question}<end_of_turn> <start_of_turn>model\n"
    return text

def test_inference(context, question):
    # text = """<start_of_turn>user
    # Explain 'AMSI init bypass' and its purpose.<end_of_turn>
    # <start_of_turn>model"""
    text = formatting_func(context, question)
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=50, 
        eos_token_id=eos_token,  # Explicitly set the EOS token ID
        num_return_sequences=1  # Ensure only one sequence is returned for clarity
    )
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    only_prompt = text.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
    return result[len(only_prompt):].strip()
    # prompt = pipe.tokenizer.apply_chat_template([{"role": "system", "content": context},{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    # outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=eos_token)
    # return outputs[0]['generated_text'][len(prompt):].strip()

eval_dataset = load_dataset("json", data_files="data/test_dataset.json", split="train")
eval_dataset = eval_dataset.shuffle().select(range(1))


for prompt in eval_dataset:
    context = prompt["messages"][0]["content"]
    user = prompt["messages"][1]["content"]
    expected_result = prompt["messages"][2]["content"]
    inference = test_inference(context, user)
    print(f"User: {user}")
    print(f"expected_result: {expected_result}")
    print(f"inference: {inference}")
    # print(f"    prompt:\n{promp}")
    # print(f"    response:\n{test_inference(prompt)}")
    print("-"*50)