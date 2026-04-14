from transformers import AutoModel
import torch.nn as nn


class ModelLayerDecomposer:
    def __init__(self, model_names):
        """
        初始化时传入 Hugging Face 模型名称列表。
        :param model_names: List[str] 模型名称列表，例如 ["bert-base-uncased", "roberta-base"]
        """
        self.model_names = model_names
        self.models = {}  # 用于存储加载后的模型对象
        self.candidate_pool = []  # 用于存储所有层信息

    def load_models(self):
        """加载所有模型并存入 self.models 字典中。"""
        for name in self.model_names:
            try:
                model = AutoModel.from_pretrained(name)
                self.models[name] = model
                print(f"模型 {name} 加载成功。")
            except Exception as e:
                print(f"加载模型 {name} 失败：{e}")

    # def _extract_layer_dimension(self, layer):
    #     """
    #     尝试提取层的维度信息。
    #     这里主要判断如果层有 weight 参数，则返回 weight 的形状；否则返回 None。
    #     :param layer: torch.nn.Module 层对象
    #     :return: 层维度信息或 None
    #     """
    #     if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
    #         return tuple(layer.weight.shape)
    #     return None

    def _extract_layer_dimension(self, layer):
        """
        尝试提取层的维度信息，根据不同的层类型返回相应的维度信息：
          - 对于 nn.Linear 层：返回 (in_features, out_features)
          - 对于 nn.Conv2d 层：返回 (in_channels, out_channels, kernel_size)
          - 对于 nn.Embedding 层：返回 (num_embeddings, embedding_dim)
          - 对于 nn.LayerNorm 层：返回 normalized_shape
          - 对于 self-attention 层：返回 (num_attention_heads, attention_head_size)
          - 其他情况：如果存在 weight 参数，则返回 weight 的 shape；否则返回 None

        :param layer: torch.nn.Module 层对象
        :return: 层维度信息或 None
        """
        import torch.nn as nn

        if isinstance(layer, nn.Linear):
            return (layer.in_features, layer.out_features)
        elif isinstance(layer, nn.Conv2d):
            # kernel_size 可能为单个整数或元组
            return (layer.in_channels, layer.out_channels, layer.kernel_size)
        elif isinstance(layer, nn.Embedding):
            return (layer.num_embeddings, layer.embedding_dim)
        elif isinstance(layer, nn.LayerNorm):
            return layer.normalized_shape
        # 针对 self-attention 层（如 Hugging Face 的 BertSelfAttention 等）
        elif hasattr(layer, 'num_attention_heads'):
            num_heads = getattr(layer, 'num_attention_heads', None)
            head_size = getattr(layer, 'attention_head_size', None)
            # 如果 attention_head_size 不存在，尝试通过 all_head_size 推导出 head_size
            if head_size is None and hasattr(layer, 'all_head_size') and num_heads:
                head_size = layer.all_head_size // num_heads
            return (num_heads, head_size)
        elif hasattr(layer, "weight") and hasattr(layer.weight, "shape"):
            return tuple(layer.weight.shape)
        else:
            return None

    def decompose(self):
        """
        遍历每个加载后的模型的所有子模块，将每一层信息保存到 candidate_pool 中。
        每个候选项是一个字典，包含以下信息：
            - model: 模型名称
            - layer_index: 层的索引（遍历顺序）
            - layer_name: 层的模块名称（在模型内部的名称）
            - layer_type: 层的类型名称
            - layer_dimension: 如果层中有 weight，则返回其形状，否则为 None
        :return: List[dict] 候选池
        """
        self.candidate_pool = []
        for model_name, model in self.models.items():
            # 使用 named_modules 遍历所有模块，跳过最顶层的模块（名称为空字符串）
            for idx, (name, module) in enumerate(model.named_modules()):
                if name == "":
                    continue  # 跳过模型根节点
                layer_info = {
                    "model": model_name,
                    "layer_index": idx,
                    "layer_name": name,
                    "layer_type": type(module).__name__,
                    "layer_dimension": self._extract_layer_dimension(module)
                }
                self.candidate_pool.append(layer_info)
        return self.candidate_pool


# # 使用示例
# if __name__ == "__main__":
#     # 你可以指定多个模型名称
#     models_to_load = ["bert-base-uncased", "roberta-base"]
#
#     decomposer = ModelLayerDecomposer(models_to_load)
#     decomposer.load_models()  # 加载模型
#     pool = decomposer.decompose()  # 获取候选池
#
#     # 打印候选池中的前几个层信息
#     for layer in pool[:10]:
#         print(layer)
