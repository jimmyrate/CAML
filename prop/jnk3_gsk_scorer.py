import warnings, numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
# 或者只屏蔽 sklearn：
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

# 兼容旧 numpy 别名
for alias, actual in {
    'int': int, 'float': float, 'bool': bool, 'object': object, 'complex': complex
}.items():
    if not hasattr(np, alias):
        setattr(np, alias, actual)

from typing import Optional
import torch
import numpy as np
from rdkit import Chem
import pickle
from rdkit.Chem import AllChem
from rdkit import DataStructs
import csv
from torch.utils.data import DataLoader
import selfies as sf
import logging
import sys
import joblib
# import sklearn
# import sklearn.ensemble._forest
# import sklearn.tree._classes
#
# # 修复老版本 sklearn 模型兼容性
# sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest
# sys.modules['sklearn.tree.tree'] = sklearn.tree._classes

class jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/home/ta/rxb/model_merging/props/jnk.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = joblib.load(f)

    def __call__(self, smiles_input):
        """
        Accept either:
          - a single SMILES string -> return a single float (np.float32)
          - an iterable of SMILES -> return np.float32 array of scores
        """
        single_input = False
        # 判断是否为单个字符串（注意：字符串本身是可迭代的，所以需要特殊处理）
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
            single_input = True
        else:
            # assume iterable of SMILES (list/tuple/pandas Series)
            smiles_list = list(smiles_input)

        fps = []
        mask = []

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                valid = int(mol is not None)
                mask.append(valid)

                if valid:
                    fp = self.fingerprints_from_mol(mol)  # should return shape (1, n_bits)
                    fp = np.asarray(fp).reshape(1, -1)  # enforce (1, n_features)
                else:
                    fp = np.zeros((1, 2048), dtype=float)
            except Exception as e:
                # 出现异常时记录为无效分子
                # 你也可以选择记录日志而不是 print
                print(f"⚠️ SMILES 处理失败: {smiles}，原因：{e}")
                mask.append(0)
                fp = np.zeros((1, 2048), dtype=float)

            fps.append(fp)

        # 合并为二维数组 (N, n_features)
        X = np.concatenate(fps, axis=0)

        # 预测 — sklearn 要求二维输入
        try:
            probs = self.clf.predict_proba(X)[:, 1]
        except Exception as e:
            # 如果 predict_proba 出错，可以尝试 predict 或返回全0
            print(f"⚠️ predict_proba 出错: {e}")
            probs = np.zeros((X.shape[0],), dtype=float)

        # mask 无效分子
        probs = probs * np.array(mask)

        probs = np.float32(probs)

        if single_input:
            # 返回单个 float（而不是数组），便于你原先逐条调用
            return float(probs[0])
        else:
            return probs

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
def get_jnk(smiles):
    jnk = jnk3_model()(smiles)
    return -jnk[0]





class toxicity_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/home/ta/rxb/model_merging/props/toxicity.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = joblib.load(f)

    def __call__(self, smiles_input):
        """
        Accept either:
          - a single SMILES string -> return a single float (np.float32)
          - an iterable of SMILES -> return np.float32 array of scores
        """
        single_input = False
        # 判断是否为单个字符串（注意：字符串本身是可迭代的，所以需要特殊处理）
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
            single_input = True
        else:
            # assume iterable of SMILES (list/tuple/pandas Series)
            smiles_list = list(smiles_input)

        fps = []
        mask = []

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                valid = int(mol is not None)
                mask.append(valid)

                if valid:
                    fp = self.fingerprints_from_mol(mol)  # should return shape (1, n_bits)
                    fp = np.asarray(fp).reshape(1, -1)  # enforce (1, n_features)
                else:
                    fp = np.zeros((1, 2048), dtype=float)
            except Exception as e:
                # 出现异常时记录为无效分子
                # 你也可以选择记录日志而不是 print
                print(f"⚠️ SMILES 处理失败: {smiles}，原因：{e}")
                mask.append(0)
                fp = np.zeros((1, 2048), dtype=float)

            fps.append(fp)

        # 合并为二维数组 (N, n_features)
        X = np.concatenate(fps, axis=0)

        # 预测 — sklearn 要求二维输入
        try:
            probs = self.clf.predict_proba(X)[:, 1]
        except Exception as e:
            # 如果 predict_proba 出错，可以尝试 predict 或返回全0
            print(f"⚠️ predict_proba 出错: {e}")
            probs = np.zeros((X.shape[0],), dtype=float)

        # mask 无效分子
        probs = probs * np.array(mask)

        probs = np.float32(probs)

        if single_input:
            # 返回单个 float（而不是数组），便于你原先逐条调用
            return float(probs[0])
        else:
            return probs

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
def get_tox(smiles):
    toxicity = toxicity_model()(smiles)
    return -toxicity[0]


class gsk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/home/ta/rxb/model_merging/props/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = joblib.load(f)

    def __call__(self, smiles_input):
        """
        Accept either:
          - a single SMILES string -> return a single float (np.float32)
          - an iterable of SMILES -> return np.float32 array of scores
        """
        single_input = False
        # 判断是否为单个字符串（注意：字符串本身是可迭代的，所以需要特殊处理）
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
            single_input = True
        else:
            # assume iterable of SMILES (list/tuple/pandas Series)
            smiles_list = list(smiles_input)

        fps = []
        mask = []

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                valid = int(mol is not None)
                mask.append(valid)

                if valid:
                    fp = self.fingerprints_from_mol(mol)  # should return shape (1, n_bits)
                    fp = np.asarray(fp).reshape(1, -1)  # enforce (1, n_features)
                else:
                    fp = np.zeros((1, 2048), dtype=float)
            except Exception as e:
                # 出现异常时记录为无效分子
                # 你也可以选择记录日志而不是 print
                print(f"⚠️ SMILES 处理失败: {smiles}，原因：{e}")
                mask.append(0)
                fp = np.zeros((1, 2048), dtype=float)

            fps.append(fp)

        # 合并为二维数组 (N, n_features)
        X = np.concatenate(fps, axis=0)

        # 预测 — sklearn 要求二维输入
        try:
            probs = self.clf.predict_proba(X)[:, 1]
        except Exception as e:
            # 如果 predict_proba 出错，可以尝试 predict 或返回全0
            print(f"⚠️ predict_proba 出错: {e}")
            probs = np.zeros((X.shape[0],), dtype=float)

        # mask 无效分子
        probs = probs * np.array(mask)

        probs = np.float32(probs)

        if single_input:
            # 返回单个 float（而不是数组），便于你原先逐条调用
            return float(probs[0])
        else:
            return probs


    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
def get_gsk3(smiles):
    gsk = gsk3_model()(smiles)
    return -gsk[0]