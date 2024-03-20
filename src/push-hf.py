import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm

peft_model_id = "./models/gemma-2b-sql-nl-it-v1"
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16)
model.push_to_hub("gemma-2b-sql-nl-it-v1")

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
tokenizer.push_to_hub("gemma-2b-sql-nl-it-v1")