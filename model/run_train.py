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
TOKENIZER_PATH = "/root/autodl-tmp/nash_merging/MolGen"  # ✅ 修改为 SELFIES tokenizer 路径
OUTPUT_MODEL_PATH = "chemgpt_pretrained_selfies/"
BATCH_SIZE = 128
MAX_LENGTH = 128
NUM_EPOCHS = 10
LR = 2e-5
MODEL_HIDDEN_SIZE = 384

# ========================
# Step 1: 加载 SELFIES tokenizer
# ========================

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/nash_merging/MolGen")
# tokenizer2 = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 通常已存在，保险起见保留
# tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>'})

# ========================
# Step 2: 定义 SELFIES 数据集
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
                formatted = " ".join(token_list)  # 只保留 token 序列，不加 BOS/EOS
                self.selfies_list.append(formatted)
            except Exception:
                continue  # 跳过非法 SMILES

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        text = self.selfies_list[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",  # 返回的是 dict，包含 input_ids 和 attention_mask
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids  # 自回归训练时通常用 input_ids 作为 labels
        }


class SelfiesDatasetFromCSV(Dataset):
    def __init__(self, selfies_list, tokenizer, max_length=512):
        """
        selfies_list: 已经是空格分隔的 SELFIES 字符串列表
        tokenizer: transformers 的 tokenizer
        max_length: 最大序列长度
        """
        self.selfies_list = selfies_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        text = self.selfies_list[idx].strip()  # 去掉可能的多余空格
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
            # 此处可以加入SMILES合法性检查
            self.smiles_list.append(smi)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        text = self.smiles_list[idx]

        # 添加起始和结束token，通常是 tokenizer.cls_token / bos_token 和 tokenizer.sep_token / eos_token
        if self.add_special_tokens:
            if self.tokenizer.bos_token is not None and self.tokenizer.eos_token is not None:
                text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            else:
                text = "<BOS>" + text + "<EOS>"  # 需要确保你的tokenizer中加入了这两个token

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
            "labels": input_ids  # 适用于自回归语言建模
        }

# ========================
# Step 3: 加载数据并转换为 SELFIES
# ========================
df = pd.read_csv(CSV_PATH)
selfies_list = df["SELFIES"].dropna().tolist()

# smiles_list = df["SMILES"].dropna().tolist()
# dataset = SmilesDataset(smiles_list, tokenizer, MAX_LENGTH)
# dataset = SelfiesDataset(smiles_list, tokenizer, MAX_LENGTH)
dataset = SelfiesDatasetFromCSV(selfies_list, tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========================
# Step 4: 初始化模型
# ========================
model = LitChemGPT(
    model_size=MODEL_HIDDEN_SIZE,
    lr=LR,
    tokenizer_dir=TOKENIZER_PATH,  # ✅ 注意传入的是 SELFIES tokenizer 路径
    logs_dir="logs",
    cache_path="."
)
# model.resize_token_embeddings(len(tokenizer))
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

print(f"🚀 模型总参数量: {total_params:,}")
print(f"✅ 可训练参数量: {trainable_params:,}")

# ========================
# Step 5: 设置回调与 Trainer
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
# Step 6: 开始训练
# ========================
seed_everything(42)
trainer.fit(model, dataloader)

# ========================
# Step 7: 保存模型与 tokenizer
# ========================
model.model.save_pretrained(OUTPUT_MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)

print(f"✅ 模型已保存至: {OUTPUT_MODEL_PATH}")
