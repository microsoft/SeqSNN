import torch.nn as nn
import torch.nn.functional as F


class VanillaAttention(nn.Module):
    def __init__(self):
        super(VanillaAttention, self).__init__()

    def forward(self, query, key, value):
        """
        Args:
            query: [batch_size, hidden_size]
            key: [batch_size, seq_len, hidden_size]
            value: [batch_size, seq_len, hidden_size]
        """
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        weight = F.softmax((query * key).sum(dim=-1, keepdim=True), dim=1)  # [batch_size, seq_len, 1]
        return (weight * value).sum(dim=1)  # [batch_size, hidden_size]
