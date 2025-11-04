import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMaskGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, num_layers: int = 3):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            channels = hidden_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出 [B,1,H,W]，范围(0,1)
        mask_logits = self.net(x)
        return torch.sigmoid(mask_logits)


def mask_regularization(mask: torch.Tensor, lambda_l1: float = 1e-3, lambda_tv: float = 1e-4, lambda_entropy: float = 0.0) -> torch.Tensor:
    # 稀疏：L1
    reg_l1 = mask.abs().mean()
    # 平滑：Total Variation（各向 TV）
    tv_h = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
    tv_w = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
    reg_tv = tv_h + tv_w
    # 置信：熵最小化（鼓励接近0或1）
    if lambda_entropy > 0:
        eps = 1e-6
        reg_entropy = -(mask * (mask + eps).log() + (1 - mask) * (1 - mask + eps).log()).mean()
    else:
        reg_entropy = mask.new_tensor(0.0)
    return lambda_l1 * reg_l1 + lambda_tv * reg_tv + lambda_entropy * reg_entropy


class DynamicMambaMaskedModel(nn.Module):
    def __init__(self, base_model: nn.Module, in_channels: int = 1, epsilon: float = 0.1,
                 mask_hidden: int = 32, mask_layers: int = 3):
        super().__init__()
        self.base_model = base_model
        self.mask_gen = DynamicMaskGenerator(in_channels=in_channels, hidden_channels=mask_hidden, num_layers=mask_layers)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        # 生成动态空间权重，并对输入进行调制
        mask = self.mask_gen(x)  # [B,1,H,W]
        x_masked = x * (self.epsilon + (1.0 - self.epsilon) * mask)
        logits = self.base_model(x_masked)
        return logits, mask


