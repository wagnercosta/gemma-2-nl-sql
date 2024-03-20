import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

model_id = "google/gemma-2b"
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
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# get token id for end of conversation
eos_token = tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"][0]

def test_inference(context, prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "system", "content": context},{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=eos_token)
    return outputs[0]['generated_text'][len(prompt):].strip()

eval_dataset = load_dataset("json", data_files="data/test_dataset.json", split="train")
eval_dataset = eval_dataset.shuffle().select(range(5))


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