# Gemma 2 - POC - Natural Language SQL Query

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