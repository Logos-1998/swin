import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --------------------------------------------------------
# ESC 核心组件 (保持原样)
# --------------------------------------------------------

def _geo_ensemble(k):
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
    return k

class LayerNorm(nn.Module):
    """支持 channels_first 的 LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.pdim = pdim
        self.lk_size = kernel_size
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)

        # 1. Dynamic Conv
        bs = x1.shape[0]
        dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
        x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
        x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size//2, groups=bs * self.pdim)
        x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)

        # 2. Large Kernel Conv
        # 注意: ESC 的 lk_filter 是 (pdim, pdim, k, k)，是 Dense Convolution
        x1 = F.conv2d(x1, lk_filter, stride=1, padding=self.lk_size // 2) + x1_

        x = torch.cat([x1, x2], dim=1)
        return x

class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim, kernel_size)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x

class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, exp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * exp_ratio)
        self.proj = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.dwc = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, kernel_size//2, groups=hidden_dim)
        self.aggr = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x

# --------------------------------------------------------
# 适配 Swin Transformer 的封装 Block
# --------------------------------------------------------

class ESCSwinBlock(nn.Module):
    """
    用来替换 SwinTransformerBlock 的 ESC 模块。
    """
    def __init__(self, dim, input_resolution, pdim, kernel_size, mlp_ratio=4.,
                 drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.pdim = pdim
        self.kernel_size = kernel_size
        self.mlp_ratio = mlp_ratio

        # [关键兼容性 1] 命名必须是 norm1 和 norm2
        # 这样 SwinV2 的 _init_respostnorm 才能正确找到它们并初始化
        self.norm1 = LayerNorm(dim, data_format="channels_first")
        self.attn = ConvAttnWrapper(dim, pdim, kernel_size)

        self.norm2 = LayerNorm(dim, data_format="channels_first")
        self.mlp = ConvFFN(dim, kernel_size=3, exp_ratio=mlp_ratio)

        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, lk_filter):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # (B, L, C) -> (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        shortcut = x

        # Block 1: ConvAttn
        x = self.norm1(x)
        x = self.attn(x, lk_filter)
        x = shortcut + self.drop_path(x)

        # Block 2: ConvFFN
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)

        # (B, C, H, W) -> (B, L, C)
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        return x

    # [关键兼容性 2] 实现 flops 方法
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        dim = self.dim
        pdim = self.pdim
        k_lk = self.kernel_size
        k_sk = 3

        # --- Block 1: ConvAttn ---
        # 1. Norm1 (Element-wise)
        flops += H * W * dim

        # 2. Dynamic Conv Generation
        # Pool (pdim*H*W) + Conv1(pdim->pdim/2) + Conv2(pdim/2->pdim*9)
        flops += H * W * pdim
        flops += pdim * (pdim // 2) + (pdim // 2) * (pdim * k_sk * k_sk)

        # 3. Dynamic Conv Application (Depthwise on pdim)
        flops += H * W * pdim * k_sk * k_sk

        # 4. Large Kernel Conv (Dense on pdim channels)
        # 这是一个 pdim -> pdim 的普通卷积，核大小 k_lk
        flops += H * W * k_lk * k_lk * pdim * pdim

        # 5. Aggr (1x1 Conv: dim -> dim)
        flops += H * W * dim * dim

        # --- Block 2: ConvFFN ---
        # 1. Norm2
        flops += H * W * dim

        # 2. Proj (1x1: dim -> hidden)
        hidden_dim = int(dim * self.mlp_ratio)
        flops += H * W * dim * hidden_dim

        # 3. DWC (3x3 depthwise)
        flops += H * W * hidden_dim * 3 * 3

        # 4. Aggr (1x1: hidden -> dim)
        flops += H * W * hidden_dim * dim

        return flops