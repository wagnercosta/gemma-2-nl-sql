import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
from random import randint
from tqdm import tqdm

peft_model_id = "./gemma-2b-sql-nl-chatml-v1"
# peft_model_id = args.output_dir

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load our test dataset
eval_dataset = load_dataset("json", data_files="data/test_dataset.json", split="train")
rand_idx = randint(0, len(eval_dataset))

# Test on sample
prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
print(f"Generated Answer:\n{outputs[0]}")
