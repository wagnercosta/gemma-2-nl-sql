from dotenv import load_dotenv
import os

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load environment variables from secrets.env
load_dotenv('secrets.env')

# Retrieve the token from environment variables
token = os.getenv('HUGGINGFACE_TOKEN')

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
print(data["train"])