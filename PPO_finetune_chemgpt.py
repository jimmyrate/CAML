import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import os
import torch
import selfies
import numpy as np
import importlib.metadata
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset
import math
from props import sascorer
import networkx as nx
# ==== 配置 ====

model_name = "/root/autodl-tmp/nash_merging/finetune_model/finetune_model/gsk3_1029_selfies"
output_dir = "/root/autodl-tmp/nash_merging/finetune_model/ppo_model/jnk3_RL"
device = "cuda" if torch.cuda.is_available() else "cpu"
from props.jnk3_gsk_scorer import *
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


MAX_LENGTH = 128
THRESHOLD = 0.8
NUM_EPOCHS = 20
GENERATE_BATCHES = 16
BATCH_SIZE = 512
gsk3_model_inst = gsk3_model()
jnk3_model_inst = jnk3_model()
tox_model_inst = toxicity_model()

# ==== 加载 tokenizer 并添加 PAD token ====
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({
            'pad_token': '<pad>',
            'bos_token': '<s>',
            'eos_token': '</s>',
        })

# ====  ====
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.resize_token_embeddings(len(tokenizer))

# ====  ====
tmp_path = "./tmp_base_model"
os.makedirs(tmp_path, exist_ok=True)
base_model.save_pretrained(tmp_path)
tokenizer.save_pretrained(tmp_path)

# ====  ====
model = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_path).to(device)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_path).to(device)

# ====  ====
class DummyDataset(TorchDataset):
    def __len__(self):
        return 1000  #
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor([tokenizer.pad_token_id])}

dummy_dataset = DummyDataset()

# ====  ====
ppo_config = PPOConfig(
    batch_size=512,
    mini_batch_size=32,
    learning_rate=5e-6,
    ppo_epochs=6,

    early_stopping=False,

    adap_kl_ctrl=True,
    init_kl_coef=0.15,
    target_kl=0.10,
)

# ====  ====
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=ppo_config,
    dataset=dummy_dataset
)


def penalized_logp(s):
    if s is None:
        return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max(len(j) for j in cycle_list)
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle

# ====  ====
def selfies_to_smiles(sf):
    sf = sf.replace(" ", "")
    try:
        smiles = selfies.decoder(sf)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return smiles
    except:
        return None

def is_valid_mol(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


def get_qed_reward(smiles_list, target_threshold=0.7):
    rewards = []
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            rewards.append(torch.tensor(0.0, dtype=torch.float32))
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Invalid mol")
            qed_val = QED.qed(mol)


            reward = 1 / (1 + math.exp(-12 * (qed_val - target_threshold)))

            rewards.append(torch.tensor(reward, dtype=torch.float32))
        except Exception as e:
            print(f"[QED Calculation Error] SMILES: {smi} | Error: {str(e)}")
            rewards.append(torch.tensor(0.0, dtype=torch.float32))
    return rewards


def get_penalized_logp_reward(smiles_list, target_threshold=2.5, clip_value=5.0):
    rewards = []
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            rewards.append(0.0)
            continue
        try:
            score = penalized_logp(smi)
            reward = 1 / (1 + math.exp(-2 * (score - target_threshold)))
            rewards.append(reward)
        except Exception as e:
            print(f"[Penalized logP Error] SMILES: {smi} | Error: {str(e)}")
            rewards.append(0.0)


    rewards = np.array(rewards, dtype=np.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # z-score
    rewards = np.clip(rewards, -clip_value, clip_value)


    return [torch.tensor(r, dtype=torch.float32) for r in rewards]


def get_sa_reward_focused(smiles_list, target_sa=2.2, invalid_penalty=-1.0,
                          max_sa=6.0, use_progressive_scaling=True, training_step=0):
    rewards = []
    sa_values = []
    valid_count = 0

    for smi in smiles_list:

        if not smi or not isinstance(smi, str):
            rewards.append(float(invalid_penalty))
            sa_values.append(None)
            continue

        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rewards.append(float(invalid_penalty))
                sa_values.append(None)
                continue


            sa_val = sascorer.calculateScore(mol)
            sa_values.append(sa_val)
            valid_count += 1


            if use_progressive_scaling:

                progress_factor = min(1.0, training_step / 5000)
                scale = 1.0 + 2.0 * progress_factor
            else:
                scale = 3.0


            if sa_val <= target_sa:

                excellence_bonus = (target_sa - sa_val) * 0.5
                base_reward = scale * (1.0 + excellence_bonus)
            else:

                decay_factor = math.exp(-(sa_val - target_sa))
                base_reward = scale * decay_factor



            final_reward = max(0.1, min(scale + 1.0, base_reward))

            rewards.append(float(final_reward))

        except Exception as e:
            print(f"[SA Calculation Error] SMILES: {smi} | Error: {e}")
            rewards.append(float(invalid_penalty))
            sa_values.append(None)


    if valid_count > 0:
        valid_sa = [s for s in sa_values if s is not None]
        valid_rewards = [r for r in rewards if r > invalid_penalty]

        avg_sa = np.mean(valid_sa)
        avg_reward = np.mean(valid_rewards)
        min_sa = min(valid_sa)

        print(f"SA Focus Stats: Valid {valid_count}/{len(smiles_list)}, "
              f"Avg SA: {avg_sa:.3f}, Min SA: {min_sa:.3f}, "
              f"Avg Reward: {avg_reward:.3f}")

    return rewards, sa_values



def get_gsk_reward(smiles_list, target_threshold=0.7):
    # rewards = []

    gsk3_scores = gsk3_model_inst(smiles_list)
    return gsk3_scores

def get_tox_reward(smiles_list, target_threshold=0.7):
    # rewards = []

    tox_scores = tox_model_inst(smiles_list)
    return 1 - tox_scores


def get_jnk_reward(smiles_list, threshold=None,sharpness=8.0):
    # rewards = []

    jnk_scores = jnk3_model_inst(smiles_list)
    # if threshold is None:

    #     threshold = np.percentile(jnk_scores, 75)
    # r = 1.0 / (1.0 + np.exp(-sharpness * (jnk_scores - 0.5)))

    # r = r - r.mean()
    return jnk_scores





# ====  ====


for epoch in range(NUM_EPOCHS):
    all_rewards = []

    for _ in tqdm(range(GENERATE_BATCHES), desc=f"Epoch {epoch + 1}"):
        prompts = [""] * BATCH_SIZE
        input_encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids = input_encodings.input_ids.to(device)
        attention_mask = input_encodings.attention_mask.to(device)

        gen_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # max_new_tokens=128,
            max_length=128,
            do_sample=True,
            top_k=50,
            top_p=0.90,
            temperature=1,
            # bad_words_ids=get_invalid_tokens(),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
        smiles_list = []
        gen_sequences = gen_output.sequences
        gen_texts = tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)



        smiles_list = [selfies_to_smiles(sf.strip()) for sf in gen_texts]
        # smiles_list = [sf.replace(" ", "") for sf in gen_texts]
        # rewards = get_qed_reward(smiles_list)
        # rewards = get_penalized_logp_reward(smiles_list)
        # gsk_rewards = get_gsk_reward(smiles_list)
        rewards = get_jnk_reward(smiles_list)
        # rewards = get_tox_reward(smiles_list)
        # alpha = 1
        # rewards = jnk_rewards - alpha * gsk_rewards
        # rewards, sa_vals = get_sa_reward_focused(smiles_list)
        # rewards, jnk_scores = get_jnk_reward_simple_linear(smiles_list)
        # rewards, jnk_scores = get_jnk_reward_precise(
        #     smiles_list,
        #     strategy="balanced",
        #     debug=True
        # )
        # for i in range(min(6, len(sa_vals))):
        #     print(f"sample {i}: SMILES={smiles_list[i]!r}, SA={sa_vals[i]}, reward={rewards[i]:.4f}")
        # rewards = np.array([float(r) for r in rewards])

        # r_min, r_max = rewards.min(), rewards.max()
        # rewards = (rewards - r_min) / (r_max - r_min + 1e-8)
        # rewards = 0.5 + 1.5 * (rewards - 0.5)
        # rewards = np.clip(rewards, 0.0, 1.0)


        query_tensors = []
        response_tensors = []
        reward_tensors = []

        # for i in range(BATCH_SIZE):
        #     response = gen_sequences[i, input_ids.shape[1]:]
        #     if response.ndim == 0 or response.numel() == 0:
        #         continue
        #     query_tensors.append(input_ids[i].squeeze())
        #     response_tensors.append(response)
        #     reward_tensors.append(rewards[i].to(device))




        # reward_tensors = [torch.tensor(r).to(device) for r in rewards]
        reward_tensors = [torch.tensor(r, dtype=torch.float32, device=device) for r in rewards]
        # reward_tensors = [r.view(1).to(device) for r in rewards]

        # query_tensors = [input_ids[i].squeeze() for i in range(BATCH_SIZE)]
        query_tensors = [input_ids[i, :] for i in range(BATCH_SIZE)]
        # response_tensors = [gen_sequences[i, input_ids.shape[1]:] for i in range(BATCH_SIZE)]
        response_tensors = [gen_sequences[i, input_ids.shape[1]:].view(-1) for i in range(BATCH_SIZE)]

        
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        all_rewards.extend(rewards)

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    print(f"Epoch {epoch + 1} mean award: {avg_reward:.4f}")

# ====  ====
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"PPO model save to: {output_dir}")
