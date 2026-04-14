import torch
import pandas as pd
import selfies as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rdkit import Chem
import torch, random, numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


model_path = "/home/ta/rxb/model_merging/finetune_model/new_model/gsk3_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token_path = "/home/ta/rxb/model_merging/model/chemgpt_pretrained_selfies"


tokenizer = AutoTokenizer.from_pretrained(token_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()


if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# ======== param ========
num_molecules = 10000
batch_size = 1000
max_length = 128
prompt = ""
seed = 42
# ========  ========
all_smiles = []

num_batches = (num_molecules + batch_size - 1) // batch_size
print(f"{num_batches} ")

for _ in tqdm(range(num_batches), desc="generation"):
    current_batch_size = min(batch_size, num_molecules - len(all_smiles))

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    input_ids = input_ids.repeat(current_batch_size, 1)
    attention_mask = attention_mask.repeat(current_batch_size, 1)
    # model.generation_config.seed = 42
    if seed is not None:
        set_seed(seed)
        print(f"[Debug] Generator seed reset to {seed}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for d in decoded:
        selfies_str = d.replace(" ", "")  # del ""
        if not selfies_str:
            continue
        try:
            smiles = sf.decoder(selfies_str)
            if not smiles:
                continue
            # 验证 SMILES 是否合法
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            all_smiles.append(smiles)

            # 如果达到目标数量，就提前结束
            if len(all_smiles) >= num_molecules:
                break
        except Exception:
            continue

    if len(all_smiles) >= num_molecules:
        break

# ======== ========
df = pd.DataFrame({"SMILES": all_smiles[:num_molecules]})
output_path = "/home/ta/rxb/model_merging/result_analysis/gsk3_new_0323.csv"
df.to_csv(output_path, index=False)

print(f"✅ complete generation {len(df)} valid molecular. save path is：{output_path}")