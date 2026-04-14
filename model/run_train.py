import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import selfies as sf
from tqdm import tqdm
from lit_chemgpt import LitChemGPT  # 模型文件保持不变

# ========================
# 参数设定
# ========================
CSV_PATH = "/root/autodl-tmp/nash_merging/dataset/filtered_selfies_dataset_pretrain.csv"
TOKENIZER_PATH = "/root/autodl-tmp/nash_merging/MolGen"
OUTPUT_MODEL_PATH = "chemgpt_pretrained_selfies/"
BATCH_SIZE = 128
MAX_LENGTH = 128
NUM_EPOCHS = 10
LR = 2e-5
MODEL_HIDDEN_SIZE = 384

# ========================

# ========================

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/nash_merging/MolGen")
# tokenizer2 = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>'})

# ========================

# ========================
class SelfiesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=512):
        self.selfies_list = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for smi in tqdm(smiles_list):
            try:
                sf_string = sf.encoder(smi)
                token_list = sf.split_selfies(sf_string)
                formatted = " ".join(token_list)
                self.selfies_list.append(formatted)
            except Exception:
                continue

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        text = self.selfies_list[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }


class SelfiesDatasetFromCSV(Dataset):
    def __init__(self, selfies_list, tokenizer, max_length=512):

        self.selfies_list = selfies_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        text = self.selfies_list[idx].strip()
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=512, add_special_tokens=True):
        self.smiles_list = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

        for smi in tqdm(smiles_list):

            self.smiles_list.append(smi)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        text = self.smiles_list[idx]


        if self.add_special_tokens:
            if self.tokenizer.bos_token is not None and self.tokenizer.eos_token is not None:
                text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            else:
                text = "<BOS>" + text + "<EOS>"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

# ========================

# ========================
df = pd.read_csv(CSV_PATH)
selfies_list = df["SELFIES"].dropna().tolist()

# smiles_list = df["SMILES"].dropna().tolist()
# dataset = SmilesDataset(smiles_list, tokenizer, MAX_LENGTH)
# dataset = SelfiesDataset(smiles_list, tokenizer, MAX_LENGTH)
dataset = SelfiesDatasetFromCSV(selfies_list, tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========================

# ========================
model = LitChemGPT(
    model_size=MODEL_HIDDEN_SIZE,
    lr=LR,
    tokenizer_dir=TOKENIZER_PATH,
    logs_dir="logs",
    cache_path="."
)
# model.resize_token_embeddings(len(tokenizer))
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

print(f"🚀 Total number of model parameters: {total_params:,}")
print(f"✅ Number of trainable parameters: {trainable_params:,}")

# ========================

# ========================
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    save_last=True,
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_weights_only=True,
)

trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=[checkpoint_callback],
    logger=True,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# ========================

# ========================
seed_everything(42)
trainer.fit(model, dataloader)

# ========================

# ========================
model.model.save_pretrained(OUTPUT_MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)

print(f"✅ model save to: {OUTPUT_MODEL_PATH}")
