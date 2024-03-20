import torch

#### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import AutoPeftModelForCausalLM

trained_dir = "models/gemma-2b-sql-nl-it-v1"
# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    trained_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(trained_dir,safe_serialization=True, max_shard_size="2GB")
merged_model.push_to_hub("gemma-2b-sql-nl-it-v1")
