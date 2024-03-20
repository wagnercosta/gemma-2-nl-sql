from dotenv import load_dotenv
import os
from huggingface_hub import login
from datasets import load_dataset

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

dataset = dataset.train_test_split(test_size=0.8)

dataset["train"].to_json("data/train_dataset.json")
dataset["test"].to_json("data/test_dataset.json")
