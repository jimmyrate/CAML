import os
import pandas as pd
import selfies
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from tqdm import tqdm
import torch

os.environ["WANDB_DISABLED"] = "true"

# ==== ====
MODEL_NAME = "/root/autodl-tmp/nash_merging/model/chemgpt_pretrained_selfies"
CSV_FILE = "/root/autodl-tmp/nash_merging/dataset/filtered/four_objective.csv"
SMILES_COLUMN = "SMILES"
OUTPUT_DIR = "/root/autodl-tmp/nash_merging/finetune_model/finetune_model/four_objective_selfies_0101"
MAX_LENGTH = 128
EPOCHS = 2
BATCH_SIZE = 128
SELFIES = True

# ==== ====
class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait_count = 0
        self.best_loss = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_loss = logs.get("loss")
        if current_loss is None:
            return
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                print(f"\nEarly stopping triggered at step {state.global_step}.\n")
                control.should_training_stop = True

# ==== ====
assert os.path.exists(CSV_FILE), f"CSV isn't available: {CSV_FILE}"

df = pd.read_csv(CSV_FILE)
assert SMILES_COLUMN in df.columns, f"clom {SMILES_COLUMN} isn't available"

def safe_smiles_to_selfies(smiles):
    try:
        return selfies.encoder(smiles)
    except:
        return None

if SELFIES:
    df["selfies"] = df[SMILES_COLUMN].apply(safe_smiles_to_selfies)
    df = df.dropna(subset=["selfies"])
    print(f" {len(df)}  SELFIES molecular")
    dataset = Dataset.from_dict({"text": df["selfies"].tolist()})
else:
    dataset = Dataset.from_dict({"text": df[SMILES_COLUMN].tolist()})

# ==== ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


tokenizer.add_special_tokens({
            'pad_token': '<pad>',
            'bos_token': '<s>',
            'eos_token': '</s>',
        })

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# ====  ====
# def tokenize_function(examples):
#     bos = tokenizer.bos_token or "<BOS>"
#     eos = tokenizer.eos_token or "<EOS>"
#     texts = [bos + t + eos for t in examples["text"]]
#     return tokenizer(
#         texts,
#         truncation=True,
#         padding="max_length",
#         max_length=MAX_LENGTH
#     )

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ====  ====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=1e-5,
    warmup_ratio=0.05,
    weight_decay=0.05,
    logging_steps=100,
    save_steps=500,
    save_total_limit=1,
    greater_is_better=False,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available()
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ==== ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    # callbacks=[CustomEarlyStoppingCallback(patience=3, min_delta=0.001)]
)

print("begin...")
trainer.train()

# ====  ====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"model save to：{OUTPUT_DIR}")
