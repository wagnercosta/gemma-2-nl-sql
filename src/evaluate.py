import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm

peft_model_id = "./models/gemma-2b-sql-nl-it-v1"
# peft_model_id = "wagnercosta/gemma-2b-sql-nl-it-v1"
# tokenizer_id = "google/gemma-2b-it"


# Load Model with PEFT adapter
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
eos_token = tokenizer("<end_of_turn>",add_special_tokens=False)["input_ids"][0]

def formatting_func(context, question):
    text = f"<start_of_turn>user\nCONTEXT:{context}\nQUESTION:{question}<end_of_turn> <start_of_turn>model\n"
    return text

def test_inference(context, question):
    text = formatting_func(context, question)
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=50, 
        eos_token_id=eos_token,  # Explicitly set the EOS token ID
        num_return_sequences=1  # Ensure only one sequence is returned for clarity
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    only_prompt = text.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
    return result[len(only_prompt):].strip()

def evaluate(sample):
    context = sample["context"]
    user = sample["question"]
    expected_result = sample["answer"]
    inference = test_inference(context, user)
    
    if inference == expected_result:
        return 1
    else:
        return 0

success_rate = []
number_of_eval_samples = 20

eval_dataset = load_dataset("json", data_files="data/test_dataset.json", split="train")
# iterate over eval dataset and predict
for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
    success_rate.append(evaluate(s))

# compute accuracy
accuracy = sum(success_rate)/len(success_rate)

print(f"Accuracy: {accuracy*100:.2f}%")