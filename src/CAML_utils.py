import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel  # 用于加载 Hugging Face 模型
import random
import numpy as np
import pandas as pd
import selfies as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rdkit import Chem
import copy
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from rdkit.Chem import QED
from props import sascorer
import networkx as nx
from rdkit.Chem import Descriptors
import math
import torch.nn.functional as F
from rdkit import rdBase
from props.jnk3_gsk_scorer import *
# import ot

rdBase.DisableLog('rdApp.warning')

gsk3_model_inst = gsk3_model()
jnk3_model_inst = jnk3_model()
toxicity_model_inst = toxicity_model()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for _, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    return sorted_reference_dict

def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def normalize(x, dim=0, eps=1e-12):
    min_vals, _ = torch.min(x, dim=dim, keepdim=True)
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    return (x - min_vals) / (max_vals - min_vals + eps)

def clamp(x, min_ratio=0.0, max_ratio=0.0):
    if len(x.size()) == 1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min_val = sorted_x[int(d * min_ratio)]
        max_val = sorted_x[int(d * (1 - max_ratio) - 1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min_val = sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max_val = sorted_x[:, int(d * (1 - max_ratio) - 1)].unsqueeze(1)
    return torch.clamp(x, min_val, max_val)



def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

# def set_attr(obj, names, val):
#     if len(names) == 1:
#         setattr(obj, names[0], val)
#     else:
#         set_attr(getattr(obj, names[0]), names[1:], val)

def set_attr(obj, names, val):
    if len(names) == 1:
        # 如果是 nn.Parameter 就直接赋值，否则转成 Parameter
        if isinstance(getattr(obj, names[0], None), nn.Parameter):
            setattr(obj, names[0], nn.Parameter(val))
        else:
            setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())

    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    if hasattr(mod, "lm_head"):
        names.append("lm_head.weight")
        orig_params += (mod.lm_head.weight,)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

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


def qed_reward(smi, target=0.7):
    """QED sigmoid reward"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        val = QED.qed(mol)
        return 1 / (1 + math.exp(-12 * (val - target)))
    except:
        return 0.0


def penalized_logp_reward(smi, target=2.5):
    """penalized logP sigmoid reward"""
    try:
        score = penalized_logp(smi)  # 你已有的 penalized_logp 函数
        return 1 / (1 + math.exp(-2 * (score - target)))
    except:
        return 0.0


def sa_reward(smi, max_score=4.0):
    """合成难度奖励: 越容易合成分数越高"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        sa = sascorer.calculateScore(mol)  # 你已有的 SA scorer
        return max(0.0, 1 - sa / max_score)  # 归一化到 [0,1]
    except:
        return 0.0

def topk_values_mask(M, K=0.2, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

# ===== SVD POT ======
def compute_common_subspace(task_mats, common_dim):
    """
    输入:
        task_mats: list of [d_out, d_in] torch matrices, 每个任务的某层参数
        common_dim: 公共子空间维度 k

    输出:
        U: [d_out, common_dim] 的正交基
    """
    # 典型做法：把所有任务参数求和 → SVD → 截断前 k 个奇异向量
    W_sum = sum(task_mats)
    U, S, V = torch.linalg.svd(W_sum, full_matrices=False)
    return U[:, :common_dim]


def project_to_subspace(W, U):
    """
    投影一个参数矩阵 W 到公共子空间 U
    输入:
        W: [d_out, d_in]
        U: [d_out, k]
    输出:
        Z = U^T W  → [k, d_in]
    """
    return U.T @ W


def compute_ot_distance(proj_list):
    """
    计算 Wasserstein / Sinkhorn OT 距离
    输入:
        proj_list: list of tensors [k, dim] 代表每个任务投影后的分布
    输出:
        标量 OT_cost
    """
    # 占位：你可以用 Sinkhorn-Knopp、OTT 或 POT 来实现真正的 OT
    # 为了代码能跑，这里用 pairwise L2 平均距离作为 placeholder
    K = len(proj_list)
    cost = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            cost += torch.norm(proj_list[i] - proj_list[j])
    return cost / (K * (K - 1) / 2)




# ========= 通用 Nash Bargaining Reward =========
def get_nash_bargaining_reward(
    smiles_list,
    property_fns,
    disagreement_points=None,
    clip_value=5.0
):
    """
    smiles_list: list of SMILES
    property_fns: dict, { "qed": fn, "plogp": fn, ... }
    disagreement_points: dict, { "qed": 0.2, "plogp": 0.2, ... }
    """
    n_props = len(property_fns)
    if disagreement_points is None:
        disagreement_points = {k: 0.2 for k in property_fns}  # 默认 0.2

    # 保存每个性质的 reward
    rewards_dict = {k: [] for k in property_fns}

    # 计算每个性质的原始 reward
    for smi in smiles_list:
        for name, fn in property_fns.items():
            val = fn(smi)
            rewards_dict[name].append(val)

    # 转 numpy 做标准化
    for name in rewards_dict:
        arr = np.array(rewards_dict[name], dtype=np.float32)

        # 如果 reward 分布范围过大，做 z-score & min-max
        if arr.std() > 1e-6:
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            arr = np.clip(arr, -clip_value, clip_value)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        rewards_dict[name] = arr

    # Nash bargaining 聚合
    final_rewards = []
    for i in range(len(smiles_list)):
        prod_val = 1.0
        for name in property_fns:
            u = rewards_dict[name][i]
            d = disagreement_points.get(name, 0.2)
            prod_val *= max(u - d, 0.0)
        final_rewards.append(torch.tensor(prod_val, dtype=torch.float32))

    return final_rewards, rewards_dict  # 同时返回各属性 reward 方便调试

def reward_wrapper(smiles_list):
    return torch.tensor(reward_fn(smiles_list, props=("QED", "logP")), dtype=torch.float32)

def selfies_to_smiles(selfies):
    selfies = selfies.replace(" ", "")
    try:
        smiles = sf.decoder(selfies)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return smiles
    except:
        return None # 解码失败的设为空

def tv_to_ordereddicts(tv_flat_checks, names):
    """
    tv_flat_checks: Tensor, shape (2, n)
    names: list of parameter names, length n
    return: list of 2 OrderedDicts
    """
    n_models, n_params = tv_flat_checks.shape
    assert len(names) == n_params, "names 数量要和参数维度一致"

    dicts = []
    for i in range(n_models):
        od = OrderedDict()
        for name, value in zip(names, tv_flat_checks[i]):
            od[name] = value.clone().detach().requires_grad_(True)
        dicts.append(od)
    return dicts




def nash_bargaining(reward_matrix, eps=1e-8):
    """
    reward_matrix: (B, K) 每个样本、每个属性的奖励
    原实现返回了 mean；为了做 policy gradient，这里返回 per-sample payoff (shape [B])
    Nash bargaining payoff: Π_k max(r_k - d_k, eps)  -> 输出 log(Π) = sum_k log(...)
    """
    # reward_matrix: numpy 或 torch，尽量转换为 torch
    if not torch.is_tensor(reward_matrix):
        reward_matrix = torch.tensor(reward_matrix, dtype=torch.float32)

    # baseline d: 每个属性的最小值 (shape [K])
    d = reward_matrix.min(dim=0)[0]  # [K]
    gains = torch.clamp(reward_matrix - d, min=eps)  # (B, K)
    log_product = torch.log(gains).sum(dim=-1)  # (B,)  每个样本的 sum log
    return log_product  # 注意：返回 per-sample tensor，不再做 mean

def get_gsk3(s):
    gsk = gsk3_model()(s)
    return gsk

def get_jnk(s):
    jnk = jnk3_model()(s)
    return jnk

def get_tox(s):
    tox = toxicity_model()(s)
    return tox



def reward_fn(smiles_list, props=("qed", "plogP", "sa", "gsk3", "tox")):
    """
    Robust batch reward function.
    - Only valid RDKit-parsable smiles are used for RDKit computations.
    - get_gsk3/get_tox are called on valid_smiles and their outputs are placed back into full-length arrays.
    - invalid smiles -> reward = 0
    """

    B = len(smiles_list)
    K = len(props)
    rewards = np.zeros((B, K), dtype=np.float32)

    # ---- 1) Clean input, build valid index list ----
    valid_idx = []
    valid_smiles = []
    invalid_idx = []

    for i, s in enumerate(smiles_list):
        if s is None or (not isinstance(s, str)) or s.strip() == "":
            invalid_idx.append(i)
        else:
            # attempt minimal sanity: strip
            s_clean = s.strip()
            # do not replace with "" here; keep original string
            valid_idx.append(i)
            valid_smiles.append(s_clean)

    # quick debug counts (可注释)
    # print(f"[debug] B={B}, valid={len(valid_idx)}, invalid={len(invalid_idx)}")

    # ---- 2) Pre-allocate model outputs for full batch, fill only valid indices ----
    need_gsk = "gsk3" in props
    need_tox = "tox" in props

    batch_gsk = np.zeros(B, dtype=np.float32)  # default zeros
    batch_tox = np.zeros(B, dtype=np.float32)

    # call models only on valid_smiles and place back to batch arrays at valid_idx
    if need_gsk and len(valid_smiles) > 0:
        try:
            gsk_preds = np.array(get_gsk3(valid_smiles))  # expect shape [len(valid_smiles),]
            if gsk_preds.shape[0] != len(valid_smiles):
                # safety: if model returned scalar or wrong shape, fallback
                gsk_preds = np.zeros(len(valid_smiles), dtype=np.float32)
                print("[Warning] get_gsk3 returned unexpected shape -> zeros used for valid indices.")
            batch_gsk[np.array(valid_idx)] = gsk_preds
        except Exception as e:
            print(f"[Warning] get_gsk3 failed: {e}. Using zeros for GSK3.")
            # batch_gsk already zeros

    if need_tox and len(valid_smiles) > 0:
        try:
            tox_preds = np.array(get_tox(valid_smiles))
            if tox_preds.shape[0] != len(valid_smiles):
                tox_preds = np.zeros(len(valid_smiles), dtype=np.float32)
                print("[Warning] get_tox returned unexpected shape -> zeros used for valid indices.")
            batch_tox[np.array(valid_idx)] = tox_preds
        except Exception as e:
            print(f"[Warning] get_tox failed: {e}. Using zeros for TOX.")
            # batch_tox already zeros

    # ---- 3) RDKit-based properties computed only for valid mols ----
    mols = []
    if len(valid_smiles) > 0:
        for s in valid_smiles:
            m = Chem.MolFromSmiles(s)
            # MolFromSmiles on valid_smiles should usually succeed but double-check
            if m is None:
                mols.append(None)
            else:
                mols.append(m)

    # For index mapping: mols[i] corresponds to valid_idx[i]
    # Build arrays for values, default zeros
    # QED
    if "qed" in props:
        j = props.index("qed")
        if len(valid_idx) > 0:
            qed_vals = []
            for m in mols:
                if m is None:
                    qed_vals.append(0.0)
                else:
                    try:
                        qed_vals.append(QED.qed(m))
                    except Exception:
                        qed_vals.append(0.0)
            qed_vals = np.array(qed_vals, dtype=np.float32)
            rewards[np.array(valid_idx), j] = 1 / (1 + np.exp(-15.0 * (qed_vals - 0.74)))

    # plogP (penalized_logp expects SMILES or mol depending on your implementation)
    if "plogp" in props:
        j = props.index("plogp")
        if len(valid_idx) > 0:
            plogp_vals = []
            for s in valid_smiles:
                try:
                    plogp_vals.append(float(penalized_logp(s)))
                except Exception:
                    plogp_vals.append(0.0)
            plogp_vals = np.array(plogp_vals, dtype=np.float32)
            rewards[np.array(valid_idx), j] = 1 / (1 + np.exp(-(plogp_vals - 2.5)))

    # sa
    if "sa" in props:
        j = props.index("sa")
        if len(valid_idx) > 0:
            sa_vals = []
            for m in mols:
                if m is None:
                    sa_vals.append(10.0)  # large penalty for unparsable mol
                else:
                    try:
                        sa_vals.append(float(sascorer.calculateScore(m)))
                    except Exception:
                        sa_vals.append(10.0)
            sa_vals = np.array(sa_vals, dtype=np.float32)
            rewards[np.array(valid_idx), j] = 1 / (1 + np.exp(2.0 * (sa_vals - 2.5)))

    # gsk3 (单调递增，越大越好) - we already filled batch_gsk at valid indices
    if "gsk3" in props:
        j = props.index("gsk3")
        # use scale controllable
        scale = 15.0
        rewards[:, j] = 1 / (1 + np.exp(-scale * (batch_gsk - 0.45)))

    # tox (假定 model 输出已经是 [0,1] 或合理范围)
    if "tox" in props:
        j = props.index("tox")
        rewards[:, j] = 1 / (1 + np.exp(15.0 * (batch_tox - 0.26)))  # already zero for invalids by construction

    # ---- 4) ensure invalid indices are zero (defensive) ----
    if len(invalid_idx) > 0:
        rewards[np.array(invalid_idx), :] = 0.0

    return rewards


def kl_expert_reg(fused_logits, expert_models, inputs, mask=None):
    """
    让融合模型的输出分布接近属性专家模型
    expert_models: dict {attr: model}
    fused_logits: (B, L, V)
    """
    p_fused = F.log_softmax(fused_logits, dim=-1)
    kl_total = 0.0
    for name, expert in expert_models.items():
        with torch.no_grad():
            expert_logits = expert(**inputs).logits  # (B, L, V)
        q_expert = F.softmax(expert_logits, dim=-1)
        kl = F.kl_div(p_fused, q_expert, reduction="batchmean")
        kl_total += kl
    return kl_total


def merge_flat_vector_from_alpha(alpha_np, lambda_scale=1.0):
    # alpha clipping
    a = np.clip(alpha_np, 0.0, 1.0).astype(np.float32)
    a = torch.tensor(a, dtype=torch.float32, device=flat_ptm.device)
    # linear combination in numpy
    # fused_delta = (a[:, None] * tau_np).sum(axis=0)  # (D,)
    fused_delta = torch.sum(a[:, None] * tau_np, dim=0)
    fused = ptm_vec + torch.tensor(lambda_scale) * fused_delta
    return fused


def merge_with_alpha_per_task(alpha_np, lambda_scale=1.0):
    # alpha clipping
    a = np.clip(alpha_np, 0.0, 1.0).astype(np.float32)
    a = torch.tensor(a, dtype=torch.float32, device=flat_ptm.device)
    # linear combination in numpy
    # fused_delta = (a[:, None] * tau_np).sum(axis=0)  # (D,)
    all_checks = tau_np.to(device)
    final_signs = resolve_sign(all_checks)
    signs_expanded = final_signs.unsqueeze(0).expand_as(all_checks)
    aligned = torch.where(signs_expanded > 0, all_checks, -all_checks)

    fused_delta = torch.sum(a[:, None] * aligned, dim=0)
    fused = ptm_vec + torch.tensor(lambda_scale) * fused_delta
    return fused



def merge_with_topk_and_sign(params, task_deltas, lambda_scale=1.0):
    """
    params: np.ndarray, shape = [3K]  # α_i, k_i, φ_i 交替排列
    task_deltas: List[np.ndarray], 每个任务的参数差向量
    """
    K = len(task_deltas)
    fused = np.zeros_like(task_deltas[0])
    total_params = len(task_deltas[0])

    for i in range(K):
        alpha = np.clip(params[3*i + 0], 0, 1)
        topk_ratio = np.clip(params[3*i + 1], 0, 1)
        phi = params[3*i + 2]
        sign_mask = 1 if phi >= 0 else -1

        delta = task_deltas[i]
        k = max(1, int(topk_ratio * total_params))

        # 取前 k 个绝对值最大的参数
        idx = np.argpartition(np.abs(delta), -k)[-k:]
        mask = np.zeros_like(delta)
        mask[idx] = 1

        # 应用 sign 和缩放
        fused += alpha * sign_mask * mask * delta

    return fused * lambda_scale

def topk_values_mask(M: torch.Tensor, K=0.7, return_mask=False):
    """Keep top-k by magnitude per row. M: [N, D] or [D]"""
    if K > 1:
        K = K / 100.0
    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)
    n, d = M.shape
    # number to KEEP
    k = int(d * K)
    # convert to "keep top k" logic
    if k <= 0:
        mask = torch.zeros_like(M, dtype=torch.bool)
    else:
        # find k-th largest by magnitude per row -> use kthvalue on abs(sorted asc)
        # torch.kthvalue gives k-th smallest. To get threshold for top-k largest:
        # We want elements >= kth_largest_abs = abs_sorted[-k]
        # equivalently, compute kth = d - k + 1 smallest? Simpler: use topk
        vals, idxs = torch.topk(M.abs(), k, dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(M, dtype=torch.bool)
        # set True at idxs positions
        rows = torch.arange(n, device=M.device).unsqueeze(1).expand_as(idxs)
        mask[rows, idxs] = True

    final_mask = mask.squeeze(0) if tuple(original_shape) == tuple(M.squeeze(0).shape) else mask
    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult: torch.Tensor, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())
    if majority_sign == 0:
        # if all zero, default +1
        majority_sign = torch.tensor(1.0, device=sign_to_mult.device)
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1.0 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor: torch.Tensor):
    # Tensor: [N_tasks, D]
    sign_to_mult = torch.sign(Tensor.sum(dim=0))  # sum across tasks -> sign per param
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult  # shape (D,)


def disjoint_merge(
    Tensor: torch.Tensor,
    alphas,
    merge_func: str,
    sign_to_mult: torch.Tensor,
):
    """
    多任务符号一致化 + α 加权聚合
    Tensor: [K, D]
    alphas: [K] or None
    """
    sign_to_mult = sign_to_mult.unsqueeze(0).expand_as(Tensor)

    rows_to_keep = torch.where(sign_to_mult > 0, Tensor > 0, Tensor < 0)
    selected = Tensor * rows_to_keep.float()

    # α 加权
    if alphas is not None:
        alphas = torch.as_tensor(alphas, dtype=Tensor.dtype, device=Tensor.device)
        selected = selected * alphas[:, None]

    if merge_func == "sum":
        disjoint_aggs = torch.sum(selected, dim=0)
    elif merge_func == "mean":
        cnt = (selected != 0).sum(dim=0).float().clamp(min=1)
        disjoint_aggs = torch.sum(selected, dim=0) / cnt
    elif merge_func == "max":
        disjoint_aggs = selected.abs().max(dim=0)[0] * sign_to_mult[0]
    else:
        raise ValueError(f"merge_func={merge_func} 未定义")

    return disjoint_aggs



def disjoint_merge_wosign(
    Tensor: torch.Tensor,
    merge_func: str,
    alphas: torch.Tensor = None,
):
    """
    多任务符号一致化 + α 加权聚合
    Tensor: [K, D]
    alphas: [K] or None
    """


    # α 加权

    alphas = torch.as_tensor(alphas, dtype=Tensor.dtype, device=Tensor.device)
    selected = Tensor * alphas[:, None]

    if merge_func == "sum":
        disjoint_aggs = torch.sum(selected, dim=0)
    elif merge_func == "mean":
        cnt = (selected != 0).sum(dim=0).float().clamp(min=1)
        disjoint_aggs = torch.sum(selected, dim=0) / cnt
    else:
        raise ValueError(f"merge_func={merge_func} 未定义")

    return disjoint_aggs

# -----------------------------
# Merge wrapper using TIES
# -----------------------------
def merge_with_ties_per_task(task_deltas: torch.Tensor,
                            alphas: np.ndarray,
                             ks: np.ndarray,
                             merge_func="sum",
                             device=None):
    """
    task_deltas: torch.Tensor [K, D]
    alphas: np.ndarray [K]
    ks: np.ndarray [K]
    return fused numpy vector (D,)
    """
    if device is None:
        device = task_deltas.device
    K, D = task_deltas.shape
    all_checks = task_deltas.clone().to(device)

    # 1️⃣ 每个任务分别做 top-k
    updated_list = []
    for i in range(K):
        masked, _ = topk_values_mask(all_checks[i].unsqueeze(0), K=float(ks[i]))
        updated_list.append(masked.squeeze(0))
    updated = torch.stack(updated_list, dim=0)

    # 2️⃣ 符号统一
    final_signs = resolve_sign(updated)

    # 3️⃣ 聚合（同符号内）
    merged = disjoint_merge(updated, alphas, merge_func, final_signs)
    # 5️⃣ 再乘符号一致方向 (确保一致方向的加权)
    fused = ptm_vec + merged

    return fused


def merge_with_nash_merging(task_deltas: torch.Tensor,
                             alphas: np.ndarray,
                             ks: np.ndarray,
                             merge_func="sum",
                             device=None):
    """
    task_deltas: torch.Tensor [K, D]
    alphas: np.ndarray [K]
    ks: np.ndarray [K]
    return fused numpy vector (D,)
    """
    if device is None:
        device = task_deltas.device
    K, D = task_deltas.shape
    all_checks = task_deltas.clone().to(device)

    # 1️⃣ 每个任务分别做 top-k
    updated_list = []
    for i in range(K):
        masked, _ = topk_values_mask(all_checks[i].unsqueeze(0), K=float(ks[i]))
        updated_list.append(masked.squeeze(0))
    updated = torch.stack(updated_list, dim=0)

    # 2️⃣ 符号统一
    # final_signs = resolve_sign(updated)

    # 3️⃣ 聚合（同符号内）
    merged = disjoint_merge_wosign(updated, merge_func, alphas)
    # 5️⃣ 再乘符号一致方向 (确保一致方向的加权)
    fused = ptm_vec + merged

    return fused



def merge_with_ties_wosign_task(task_deltas: torch.Tensor,
                             # alphas: np.ndarray,
                             ks: np.ndarray,
                             merge_func="sum",
                             device=None):
    """
    task_deltas: torch.Tensor [K, D]
    alphas: np.ndarray [K]
    ks: np.ndarray [K]
    return fused numpy vector (D,)
    """
    if device is None:
        device = task_deltas.device
    K, D = task_deltas.shape
    all_checks = task_deltas.clone().to(device)

    # 1️⃣ 每个任务分别做 top-k
    updated_list = []
    for i in range(K):
        masked, _ = topk_values_mask(all_checks[i].unsqueeze(0), K=float(ks[i]))
        updated_list.append(masked.squeeze(0))
    updated = torch.stack(updated_list, dim=0)

    # 2️⃣ 符号统一
    # final_signs = resolve_sign(updated)

    # 3️⃣ 聚合（同符号内）
    # merged = disjoint_merge(updated, merge_func, final_signs,alphas)
    # merged = torch.sum(alphas[:, None] * updated, dim=0)
    merged = torch.sum(updated, dim=0)
    # 5️⃣ 再乘符号一致方向 (确保一致方向的加权)
    fused = ptm_vec + merged

    return fused


def generate_and_eval_from_flat(fused, n_mols=256, prompt="", batch_size=None, decode_fn=None,
                                props=("QED", "logP", "MW"), seed=None):
    """
    ...
    seed: int, optional. 如果提供，将在 generate 前强制固定随机种子，确保生成结果一致。
    """
    model = ptm_model
    fused_state = vector_to_state_dict(fused, ptm_check, remove_keys=[])
    model.load_state_dict(fused_state, strict=False)
    model.to(device)
    model.eval()

    if batch_size is None:
        batch_size = min(512, n_mols)

    generated_smiles = []
    n_done = 0

    # -----------------------------------------------------------
    # 【关键修改】在此处重置种子
    # 这样无论之前运行了多少轮优化，只要这里传入 seed=42，
    # 生成的分子就永远是那一批固定的分子。
    # -----------------------------------------------------------
    if seed is not None:
        set_seed(seed)
        print(f"[Debug] Generator seed reset to {seed}")

    try:
        tokenizer
    except NameError:
        raise RuntimeError("请在环境中提供 tokenizer")

    pbar = tqdm(total=n_mols, desc="Generating molecules")

    while n_done < n_mols:
        cur_batch = min(batch_size, n_mols - n_done)
        # 生成 prompt tokens
        enc = tokenizer([prompt] * cur_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_length=128,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # decode -> smiles/selfies
        n_generated = len(out)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        # 如果需要把 SELFIES 转 SMILES，请提供 decode_fn 或用 selfies.decoder
        smiles_list = [selfies_to_smiles(s.strip()) for s in decoded]
        generated_smiles.extend(smiles_list)
        # 过滤/清洗
        n_done += cur_batch
        pbar.update(n_generated)
    # 3) 评估：调用你已有的 reward_fn（需返回 shape (B, M)）
    # reward_fn 的签名在你的环境里定义为 reward_fn(smiles_list, props=...), 返回 np.ndarray (B,M)
    rewards, metrics = reward_fn(generated_smiles, props=props)  # np.ndarray (n_mols, M)
    # 取均值作为 candidate 的评分向量
    mean_scores = np.nanmean(rewards, axis=0)  # (M,)
    return mean_scores, rewards, generated_smiles, metrics