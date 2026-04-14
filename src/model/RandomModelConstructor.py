import torch
import torch.nn as nn
import random
from copy import deepcopy
from src.near_score import get_near_score, estimate_layer_size
import numpy as np

class RandomModelConstructor:
    """
    该类用于从候选层池中随机组合层构建新的模型，
    并评估模型的层兼容性、NEAR 得分以及验证集准确率，
    以便构造代理模型训练数据。
    """

    def __init__(self, pool, validation_loader, device=None):
        """
        参数：
            pool: 候选池，字典格式，每个 key 对应一个候选层，
                  例如：{'layer_0': {'model': 'model_A', 'index': 0,
                                       'module_name': 'conv1', 'type_name': 'Conv2d',
                                       'layer': <nn.Conv2d>}, ...}
            validation_loader: 验证集的数据加载器（DataLoader）
            device: 设备 "cuda" 或 "cpu"，如果为 None，将自动判断
        """
        self.pool = pool
        self.validation_loader = validation_loader
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class CombinedModel(nn.Module):
        """
        将随机选出的层按照顺序拼接构造出一个新的模型。
        """

        def __init__(self, layers):
            super(RandomModelConstructor.CombinedModel, self).__init__()
            # 使用 ModuleList 注册层，便于参数管理
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    def random_combine_models(self, num_models, num_layers_per_model):
        """
        从候选池中随机构造新模型。

        参数：
            num_models: 随机构造新模型的数量
            num_layers_per_model: 每个模型选取的层数量

        返回：
            candidate_models: 生成的新模型列表
        """
        candidate_models = []
        pool_keys = list(self.pool.keys())
        # 生成指定数量的新模型
        for i in range(num_models):
            # 随机选择不重复的层
            selected_keys = random.sample(pool_keys, num_layers_per_model)
            # 克隆选出的层实例
            selected_layers = [deepcopy(self.pool[k]['layer']) for k in selected_keys]
            new_model = self.CombinedModel(selected_layers)
            candidate_models.append(new_model)
        return candidate_models

    def evaluate_compatibility(self, model_layers):
        """
        评估模型的层兼容性，
        返回值为 0~1 之间的分数（这里使用示例固定值）。

        实际使用中，请根据各层特性、统计信息或其他指标计算。
        """
        score = 0.0
        for i in range(len(model_layers) - 1):
            layer1 = model_layers[i]['layer_obj']
            layer2 = model_layers[i + 1]['layer_obj']
            params1 = torch.cat([p.flatten() for p in layer1.parameters()])
            params2 = torch.cat([p.flatten() for p in layer2.parameters()])
            similarity = torch.cosine_similarity(params1, params2, dim=0)
            score += similarity.item()
        return score / (len(model_layers) - 1)

    def evaluate_near(self, model,train_dataloader):
        """
        评估模型的 NEAR 得分，
        返回值为 0~1 之间的分数（这里使用示例固定值）。

        实际使用中，请替换为真实计算逻辑。
        """
        # TODO: 根据模型整体表达能力、层特征多样性等计算 NEAR 值

        near_score = get_near_score(model, train_dataloader, repetitions=10, layer_index=None)

        return near_score

    def evaluate_validation_accuracy(self, model):
        """
        在验证集上评估模型的推理准确率。

        参数：
            model: 待评估的模型

        返回：
            accuracy: 在验证集上的准确率（0~1之间）
        """
        model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                # 假设输出为 (batch, classes)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def build_and_evaluate(self, num_models, num_layers_per_model):
        """
        随机构造新的模型，并对每个模型评估：
            - 层兼容性得分和 NEAR 得分加权合成的总分 X
            - 验证集推理得到的准确率 y

        参数：
            num_models: 随机构造的新模型数量
            num_layers_per_model: 每个新模型的层数

        返回：
            training_data: 列表，每个元素为字典，包含新模型以及对应的 X 和 y，
                           例如：{'model': new_model, 'X': X, 'y': y}
        """
        training_data = []
        candidate_models = self.random_combine_models(num_models, num_layers_per_model)
        for idx, model in enumerate(candidate_models):
            comp_score = self.evaluate_compatibility(model)
            near_score = self.evaluate_near(model)
            # 加权组合：各占 0.5 权重
            X = 0.5 * comp_score + 0.5 * near_score
            y = self.evaluate_validation_accuracy(model)
            training_data.append({
                'model': model,
                'X': X,
                'y': y
            })
            print(f"模型 {idx + 1}: 兼容性={comp_score}, NEAR={near_score}, X={X}, Accuracy={y}")
        return training_data

    def prepare_training_data(training_samples):
        """
        将训练样本列表转换为特征矩阵 X_train 和目标向量 y_train。

        参数：
        - training_samples: list of dict，每个字典包含 'compatibility', 'near', 'accuracy' 键

        返回：
        - X_train: numpy.ndarray，形状为 (n_samples, 2)
        - y_train: numpy.ndarray，形状为 (n_samples,)
        """
        X_train = []
        y_train = []
        for sample in training_samples:
            # 提取特征
            features = [sample['compatibility'], sample['near']]
            X_train.append(features)
            # 提取目标值
            y_train.append(sample['accuracy'])
        return np.array(X_train), np.array(y_train)

    def generate_candidates(self, num_candidates=100):
        """随机生成候选层组合（可扩展为启发式生成）"""
        candidates = []
        for _ in range(num_candidates):
            layers = random.sample(self.pool, k=self.num_layers_per_model)
            candidates.append(layers)
        return candidates

    def extract_features(self, candidate_models):
        """提取层间兼容性和near值作为特征"""
        X = []
        for layers in candidate_models:
            # 假设兼容性评分和near值已预计算并存储
            compatibility = np.mean([layer.compatibility for layer in layers])
            near_value = np.mean([layer.near for layer in layers])
            X.append([compatibility, near_value])
        return np.array(X)

    def assemble_model(self, layers):
        """根据层列表构建新模型"""
        new_model = Model()
        for layer in layers:
            new_model.add_layer(layer)
        return new_model

# 使用示例：
# if __name__ == '__main__':
#     # 假设已经通过 ModelLayerDecomposer 得到候选池 pool
#     # 如： pool = decomposer.decompose()
#     pool = {
#         # 示例：每个字典项中 'layer' 为层的实例，其他字段可供记录信息使用
#         'layer_0': {'model': 'model_A', 'index': 0, 'module_name': 'conv1', 'type_name': 'Conv2d',
#                     'layer': nn.Conv2d(3, 16, kernel_size=3, padding=1)},
#         'layer_1': {'model': 'model_B', 'index': 1, 'module_name': 'relu1', 'type_name': 'ReLU',
#                     'layer': nn.ReLU()},
#         'layer_2': {'model': 'model_C', 'index': 2, 'module_name': 'conv2', 'type_name': 'Conv2d',
#                     'layer': nn.Conv2d(16, 32, kernel_size=3, padding=1)},
#         'layer_3': {'model': 'model_A', 'index': 3, 'module_name': 'bn1', 'type_name': 'BatchNorm2d',
#                     'layer': nn.BatchNorm2d(16)},
#         # 可以继续添加更多候选层...
#     }
#
#     # 假设你已经定义好了验证集的数据加载器 validation_loader
#     # 例如：
#     # from torch.utils.data import DataLoader
#     # validation_dataset = YourValidationDataset(...)
#     # validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
#     validation_loader = ...  # 请替换为实际的 DataLoader
#
#     # 初始化随机模型构造器
#     constructor = RandomModelConstructor(pool, validation_loader)
#
#     # 随机构造 10 个新模型，每个模型由 3 个层拼接而成
#     training_samples = constructor.build_and_evaluate(num_models=10, num_layers_per_model=3)
#
#     # training_samples 为列表，每个元素包括新模型对象、X 值、y 值
