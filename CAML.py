
import numpy as np
import torch
from cma import CMAEvolutionStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from src.CAML_utils import *
from src.task_vectors import *
from functools import partial
import time
import math
import os
from tqdm import tqdm
import pandas as pd
import json
import transformers
transformers.logging.set_verbosity_error()

reset_thresh = 0.5
device = 'cuda:0'
prop_expert = ['qed', 'plogp']
model = 'chemGPT'


if model == 'chemGPT':
    pretrain_model_path = "/root/autodl-tmp/nash_merging/model/chemgpt_pretrained_selfies"
    ft_model_path = [f"/root/autodl-tmp/nash_merging/finetune_model/ppo_model/{prop_name}_RL" for prop_name in prop_expert]
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    ptm_model = AutoModelForCausalLM.from_pretrained(pretrain_model_path)
    ptm_check = AutoModelForCausalLM.from_pretrained(pretrain_model_path).to(device).state_dict()
    ft_checks = [AutoModelForCausalLM.from_pretrained(finetune_model).to(device).state_dict() for finetune_model in ft_model_path]
    check_parameterNamesMatch(ft_checks + [ptm_check])
    remove_keys = []
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm


elif model == 'molgen':
    pretrain_model_path = ""


K = tv_flat_checks.shape[0]    # 专家数
D = flat_ptm.numel()           # 参数维度
ptm_vec = flat_ptm
tau_np = tv_flat_checks  # shape (K, D), numpy 以便 cma 使用
# model = ptm_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_CAML(
        tv_flat_checks,
        generations=30,
        popsize=8,
        sigma_init=0.3,
        n_mols=1000,
        per_candidate_repeat=1,
        merge_func="sum",
        alpha_init=0.5,
        k_init=0.7,
        props=("QED", "logP"),
        prompt="",
        decode_fn=None,
        log_every=1,
        save_json="nash_merging_Regularization_history.json",
        best_model_path="best_nash_model.pt",
        best_smiles_path="best_nash_smiles.csv",
        baseline_dict=None,
        ft_model_paths=None,
        eval_model_fn=None,
        baseline_num_samples=10000,
        lambda_conflict=0.5,
        lambda_drop=1.0,
        beta_weight=3.0,
        tol_drop=0.01,
        eps=1e-8,
        compute_conflict_for_topk=None,
):
    """
    Modified to optimize on Transformed Reward but LOG Raw Metrics.
    """

    device = tv_flat_checks.device
    K = tv_flat_checks.shape[0]
    prop_list = list(props)
    P = len(prop_list)

    if baseline_dict is None:
        raise ValueError("You must provide baseline_dict...")
    else:
        for p in prop_list:
            if p not in baseline_dict:
                raise ValueError(f"baseline_dict missing prop {p}")



    baseline_scores = np.array([baseline_dict[p] for p in prop_list], dtype=float)

    mean_init = np.concatenate([
        np.full(K, k_init),
        np.full(K, alpha_init),
    ])
    bounds = [0.0, 1.0]
    es = CMAEvolutionStrategy(mean_init, sigma_init, {
        "popsize": popsize,
        "bounds": bounds,
        'seed': 42,
    })

    history = []
    best_overall = None
    best_overall_fused = None
    best_overall_smiles = []

    for gen in tqdm(range(generations), desc="SACMA-ES generations", dynamic_ncols=True):

        candidate_list = es.ask()

        fitnesses, score_vectors_raw = [], []
        gen_records = []
        gen_fused_models = []
        gen_smiles_collection = []
        candidate_infos = []

        # First pass
        for idx, cand in enumerate(
                tqdm(candidate_list, leave=False, desc=f"Gen {gen:02d} candidates", dynamic_ncols=True)
        ):
            ks = cand[:K]
            alphas = cand[K:]
            ks_t = torch.tensor(ks, dtype=torch.float32, device=device)
            alphas_t = torch.tensor(alphas, dtype=torch.float32, device=device)

            fused_flat = merge_with_nash_merging(
                tv_flat_checks,
                alphas_t,
                ks_t,
                merge_func=merge_func,
                device=device,
            )
            current_fused_flat = fused_flat.detach().cpu()

            agg_scores_list = []  # Transformed
            raw_metrics_list = []  # Raw
            scores_matrix_list = []  # Transformed matrix
            current_cand_smiles = []

            for _ in range(per_candidate_repeat):
                # 捕获 metrics (RAW)
                mean_scores, scores_matrix, generated_smiles, metrics = generate_and_eval_from_flat(
                    fused_flat,
                    n_mols=n_mols // per_candidate_repeat,
                    prompt=prompt,
                    decode_fn=decode_fn,
                    props=props,
                    seed=42
                )

                agg_scores_list.append(mean_scores)  # Transformed
                raw_metrics_list.append(metrics)  # Raw
                scores_matrix_list.append(scores_matrix)
                current_cand_smiles.extend(generated_smiles)


            agg_trans = np.mean(np.stack(agg_scores_list, axis=0), axis=0)
            scores_all = np.vstack(scores_matrix_list)


            agg_raw_vec = np.zeros_like(agg_trans)
            for i, p in enumerate(props):
                agg_raw_vec[i] = np.mean([m[p] for m in raw_metrics_list])

            candidate_infos.append({
                "idx": idx,
                "cand": cand,
                "ks": ks,
                "alphas": alphas,
                "fused_flat": current_fused_flat,
                "agg": agg_trans,
                "agg_raw": agg_raw_vec,
                "scores_all": scores_all,
            })
            gen_smiles_collection.append(current_cand_smiles)

        # Conflict calculation (use scores_all which is Transformed, correct for correlation)
        if compute_conflict_for_topk is not None and isinstance(compute_conflict_for_topk,
                                                                int) and compute_conflict_for_topk > 0:
            temp_fits = []
            for info in candidate_infos:
                a = info["agg"]
                w = np.exp(-beta_weight * a)
                w = w / (np.sum(w) + eps)
                temp_fits.append(float(np.sum(w * np.log(a + eps))))
            topk_idx = np.argsort(temp_fits)[-compute_conflict_for_topk:]
        else:
            topk_idx = None

        # Second pass: Compute Scalar Util
        for info_idx, info in enumerate(candidate_infos):
            idx = info["idx"]
            cand = info["cand"]
            ks = info["ks"]
            alphas = info["alphas"]
            agg = info["agg"]  # Transformed
            agg_raw = info["agg_raw"]  # Raw
            scores_all = info["scores_all"]

            # Nash Product using Transformed Scores (Safe)
            w = np.exp(-beta_weight * agg)
            w = w / (np.sum(w) + eps)
            weighted_log_prod = float(np.sum(w * np.log(agg + eps)))

            if topk_idx is not None and info_idx not in topk_idx:
                scalar_util = weighted_log_prod
                conflict = None
                sum_drop = None
            else:
                conflict = 0.0
                if scores_all.shape[0] >= 3:
                    try:
                        R = np.corrcoef(scores_all.T)
                        mask_neg = (R < 0)
                        if np.any(mask_neg): conflict = float(np.mean(np.abs(R[mask_neg])))
                    except:
                        pass
                else:
                    cov = np.cov(scores_all.T) if scores_all.shape[0] > 1 else np.zeros((P, P))
                    mask_neg = (cov < 0)
                    conflict = float(np.mean(np.abs(cov[mask_neg])) if np.any(mask_neg) else 0.0)


                drops = np.maximum(0.0, baseline_scores - agg - tol_drop)
                sum_drop = float(np.sum(drops))
                scalar_util = weighted_log_prod - lambda_conflict * conflict - lambda_drop * sum_drop

            gen_records.append({
                "gen": gen,
                "cand_idx": idx,
                "ks": ks.tolist(),
                "alphas": alphas.tolist(),
                "scalar_util": float(scalar_util),
                "scores": {prop_list[i]: float(agg_raw[i]) for i in range(P)},  # Save RAW
                "conflict": None if conflict is None else float(conflict),
                "sum_drop": None if sum_drop is None else float(sum_drop),
            })

            fitnesses.append(float(scalar_util))
            score_vectors_raw.append(agg_raw.tolist())  # Save RAW
            gen_fused_models.append(info["fused_flat"])

        es.tell(candidate_list, [-f for f in fitnesses])

        best_idx = int(np.argmax(fitnesses))
        best_fit = float(fitnesses[best_idx])
        best_cand = candidate_list[best_idx]
        best_ks = best_cand[:K]
        best_alphas = best_cand[K:]
        best_scores_raw = score_vectors_raw[best_idx]

        history.append({
            "gen": gen,
            "sigma": es.sigma,
            "candidates": gen_records,
            "best_fit": best_fit,
            "best_ks": best_ks.tolist(),
            "best_alphas": best_alphas.tolist(),
            "best_scores": best_scores_raw,  # Save RAW
        })

        if gen % log_every == 0:
            print(f"[Gen {gen:03d}] best_fit={best_fit:.4f}")
            print(f"  k={np.round(best_ks, 3)}")
            print(f"  alpha={np.round(best_alphas, 3)}")
            print(f"  Real Scores={np.round(best_scores_raw, 4)}")  # Print RAW
            print(f"  σ={es.sigma:.4f}\n")

        if best_overall is None or best_fit > best_overall["best_fit"]:
            best_overall = {
                "gen": gen,
                "best_fit": best_fit,
                "best_ks": best_ks.tolist(),
                "best_alphas": best_alphas.tolist(),
                "best_scores": best_scores_raw,  # Save RAW
            }
            best_overall_fused = gen_fused_models[best_idx]
            best_overall_smiles = gen_smiles_collection[best_idx]

        if es.stop():
            break

    with open(save_json, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saving best model to {best_model_path}...")
    best_state_dict = vector_to_state_dict(best_overall_fused, ptm_check, remove_keys=remove_keys)
    torch.save(best_state_dict, best_model_path)
    print("Model saved.")

    if best_overall_smiles:
        print(f"Saving best SMILES to {best_smiles_path}...")
        df_smiles = pd.DataFrame({"smiles": best_overall_smiles})
        df_smiles.to_csv(best_smiles_path, index=False)
        print("SMILES saved.")

    return best_overall, history


if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    print(f"Global seed set to {SEED}")
    baseline_dict = {
        "sa": 0.5,
        "plogp":0.5,
        "qed": 0.5,
        "gsk3": 0.5,
        "tox": 0.5,
    }
    best, hist = run_CAML(
        tv_flat_checks=tv_flat_checks,
        generations=30,
        popsize=15,
        sigma_init=0.3,
        n_mols=5000,
        per_candidate_repeat=1,
        merge_func="sum",
        alpha_init=0.5,
        k_init=0.7,
        props=('qed', 'plogp'),
        prompt="",
        decode_fn=None,
        baseline_dict=baseline_dict
    )

