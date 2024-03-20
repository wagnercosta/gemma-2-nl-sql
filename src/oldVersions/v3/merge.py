import torch

#### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import AutoPeftModelForCausalLM

output_dir = "microsoft-phi2-text-to-sql"
# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_dir,safe_serialization=True, max_shard_size="2GB")
