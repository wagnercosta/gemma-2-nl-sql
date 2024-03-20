# Fine-tuning Gemma 2 for Natural Language to SQL (POC)

## Introduction

Creating python environment for the project:

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```


```bash
chmod +x install.sh
```

```bash
./install.sh
```

Create a `secrets.env` file with the following content:

```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## Running the project

The following script will create the dataset and train the model (inside `data` folder):

```bash
python src/create-dataset.py
```

The following script will train (fine-tune) the model:

```bash
python src/train.py
```

The following script will merge the model:

```bash
python src/merge.py
```

To make a small test, run the following script:

```bash
python src/test.py
```

To evaluate the model, run the following script:

```bash
python src/evaluate.py
```

## Evaluation Results

This project is just a small proof of concept, before creating a similar project to my UNESP IA course final project.
Here the training arguments:

```python
args = TrainingArguments(
    output_dir="models/gemma-2b-sql-nl-it-v1", # directory to save and repository id
    num_train_epochs=5,                     # number of training epochs
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
```

The model was fine-tuned using 4000 examples of SQL queries and natural language questions, and evaluated using 1000 examples.

The evaluation results are a simple match between the predicted SQL query and the expected SQL query (this approach 
is not the best, but it is a simple way to evaluate the model)

Here the evaluation results with 1000 samples (see `src/evaluate.py` for details):

```
Accuracy: 59.20%
```