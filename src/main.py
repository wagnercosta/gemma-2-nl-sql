from dotenv import load_dotenv
import os
from huggingface_hub import login
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format, SFTTrainer
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

# Load environment variables from secrets.env
load_dotenv('secrets.env')

# Retrieve the token from environment variables
token = os.getenv('HUGGINGFACE_TOKEN')

login(
  token=token, # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

# Load dataset from the hub
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle().select(range(5000))

def formatting_func(example):
    output_texts = []
    for i in range(len(example['context'])):
        text = f"<start_of_turn>user\nCONTEXT:{example['context'][i]}\nQUESTION:{example['question'][i]}<end_of_turn> <start_of_turn>model\n{example['answer'][i]}<end_of_turn>"
        output_texts.append(text)
    return output_texts

# Load jsonl data from disk
# dataset = load_dataset("json", data_files="data/train_dataset.json", split="train")
# dataset = dataset.shuffle().select(range(5000))

# split dataset into 10,000 training samples and 2,500 test samples
# dataset = dataset.train_test_split(test_size=2500/12500)

# Hugging Face model id
model_id = "google/gemma-2b-it"
tokenizer_id = "philschmid/gemma-tokenizer-chatml"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir="models/gemma-2b-sql-nl-it-v1", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

max_seq_length = 1512 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    },
    formatting_func=formatting_func
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model()

# free the memory again
del model
del trainer
torch.cuda.empty_cache()