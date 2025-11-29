import math
import torch
from torch import nn
from torchinfo import summary



class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding module"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        time_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        time_emb = time_emb.view(*shape, self.dim)
        return time_emb


class Upsample(nn.Module):
    """Upsample features module"""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    """Downsample features module"""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim, 
                              out_channels=dim, 
                              kernel_size=3,
                              stride=2,
                              padding=1)

    def forward(self, x):
        return self.conv(x)
    

class Block(nn.Module):
    """Basic component of ResNet block"""
    def __init__(self, 
                 dim: int, 
                 dim_out: int, 
                 groups: int = 32, 
                 dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(num_groups=groups, num_channels=dim),
                                   nn.SiLU(),
                                   nn.Dropout2d(dropout) if dropout != 0 else nn.Identity(),
                                   nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1))

    def forward(self, x):
        return self.block(x)
    

class ResNetBlock(nn.Module):
    """ResNet block with time embedding"""
    def __init__(self, 
                 dim: int, 
                 dim_out: int,
                 time_dim: int, 
                 groups: int = 32, 
                 dropout: float = 0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim_out*2)
        )        
        self.block1 = Block(dim, dim_out, groups=groups, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=1) \
            if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_embed=None):
        h = self.block1(x)
        if time_embed is not None:
            gamma, beta = self.mlp(time_embed).view(time_embed.shape[0], -1, 1, 1).chunk(2, dim=1)
            h = (1 + gamma) * h + beta
        h = self.block2(h)
        return h + self.res_conv(x)
    

class SelfAttention(nn.Module):
    """Self-attention module"""
    def __init__(self, 
                 in_channel: int, 
                 n_head: int = 1, 
                 groups: int = 32):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(groups, in_channel)
        self.qkv = nn.Conv2d(in_channels=in_channel, out_channels=3*in_channel, kernel_size=1, bias=False)
        self.out = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)

    def forward(self, x):
        batch_size, channel, height, width = x.shape
        head_dim = channel // self.n_head

        norm = self.norm(x)
        qkv = self.qkv(norm).view(batch_size, self.n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch_size, self.n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch_size, self.n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch_size, channel, height, width))
        return out + x 
    

class ResNetAttnBlock(nn.Module):
    """ResNet plus self-attention block"""
    def __init__(self, 
                 dim: int, 
                 dim_out: int, 
                 time_dim: int,
                 groups: int = 32, 
                 dropout: float = 0,
                 with_attn: bool = False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResNetBlock(dim, dim_out, time_dim, groups, dropout)
        self.attn = SelfAttention(dim_out, groups=groups)

    def forward(self, x, time_embed=None):
        x = self.res_block(x, time_embed)
        if self.with_attn:
            x = self.attn(x)
        return x
    

class UNet(nn.Module):
    """U-Net denoiser

    A U-Net architecture used as the denoising backbone in diffusion models.
    The network consists of a downsampling path, middle bottleneck blocks,
    and an upsampling path with skip connections. It supports temporal
    conditioning via time embeddings and optional spatial self-attention.

    Args:
        height (int): 
            The initial input height (and width, assuming square input) of the feature map.
            Used to determine which layers include attention and to compute resolution changes
            during downsampling and upsampling.

        in_channel (int): 
            Number of input channels (e.g., 3 for RGB images, or 6 if concatenating conditions).

        out_channel (int): 
            Number of output channels of the final convolution layer (usually same as input).

        time_dim (int): 
            Dimensionality of the time embedding vector, which conditions each block
            on the diffusion timestep or other temporal variable.

        groups (int): 
            Number of groups used in Group Normalization inside each block.

        dropout (float): 
            Dropout rate applied within residual blocks to prevent overfitting.

        mults (tuple): 
            Channel expansion multipliers for each downsampling stage.
            For example, `(1, 2, 4, 8)` means that channels are scaled by these factors
            relative to `time_dim` at each level.

        num_blocks (int): 
            Number of residual-attention blocks (`ResNetAttnBlock`) per resolution level.

        with_attn_height (tuple): 
            A tuple of spatial resolutions (heights) at which self-attention
            is applied within the network. Typically includes smaller spatial sizes
            (e.g., `(16, 8)`) to capture global context when feature maps are compact.
    """

    def __init__(
            self,
            height: int,
            in_channel: int,
            out_channel: int, 
            time_dim: int,
            groups: int,
            dropout: float,
            mults: tuple,
            num_blocks: int,
            with_attn_height: tuple
    ):
        super().__init__()

        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # hyperparameters for UNet
        num_mults = len(mults)
        pre_channel = time_dim
        feat_channel = [pre_channel]

        # Downsampling layers
        down = [nn.Conv2d(in_channel, time_dim, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            aft_channel = mults[ind] * time_dim
            use_attn = (height in with_attn_height)

            for _ in range(num_blocks):
                down.append(ResNetAttnBlock(pre_channel, 
                                            aft_channel, 
                                            time_dim,
                                            groups, 
                                            dropout,
                                            with_attn=use_attn))
                feat_channel.append(aft_channel)
                pre_channel = aft_channel

            if ind != num_mults - 1:
                down.append(Downsample(pre_channel))
                feat_channel.append(pre_channel)
                height = height // 2
        self.down = nn.ModuleList(down)

        # Middle layers
        self.mid = nn.ModuleList([
            ResNetAttnBlock(pre_channel, 
                            pre_channel, 
                            time_dim,
                            groups, 
                            dropout,
                            with_attn=True),
            ResNetAttnBlock(pre_channel, 
                            pre_channel, 
                            time_dim,
                            groups, 
                            dropout,
                            with_attn=False)
            ])

        # Upsampling layers
        up = []
        for ind in reversed(range(num_mults)):
            aft_channel = mults[ind] * time_dim
            use_attn = (height in with_attn_height)

            for _ in range(0, num_blocks + 1):
                up.append(ResNetAttnBlock(pre_channel + feat_channel.pop(), 
                                          aft_channel, 
                                          time_dim,
                                          groups, 
                                          dropout,
                                          with_attn=use_attn))
                pre_channel = aft_channel

            if ind != 0:
                up.append(Upsample(pre_channel))
                height = height * 2
        self.up = nn.ModuleList(up) 

        # final convolution layer
        self.final_conv = Block(pre_channel, out_channel, groups)

    def forward(self, x, time=None):
        time_embed = self.time_mlp(time) if time is not None else None

        feats = []
        for layer in self.down:
            if isinstance(layer, ResNetAttnBlock):
                x = layer(x, time_embed)
            else: 
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResNetAttnBlock):
                x = layer(x, time_embed)
            else:
                x = layer(x)

        for layer in self.up:
            if isinstance(layer, ResNetAttnBlock):
                # 获取跳跃连接特征
                skip_feat = feats.pop()
                
                # 确保空间尺寸匹配
                if skip_feat.shape[-2:] != x.shape[-2:]:
                    # 使用插值调整skip特征的尺寸以匹配x
                    skip_feat = torch.nn.functional.interpolate(
                        skip_feat, 
                        size=x.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                x = layer(torch.cat((x, skip_feat), dim=1), time_embed)
            else:
                x = layer(x)

        return self.final_conv(x)



if __name__ == "__main__":
    model = UNet(
            height=64,
            in_channel=3,
            out_channel=3,
            time_dim=128,
            groups=32,
            dropout=0.1,
            mults=(1, 2, 4, 8),
            num_blocks=3,
            with_attn_height=(16, 8)
    )
    x = torch.randn(1, 3, 64, 64)
    t = torch.randn(1)
    out = model(x, t)
    summary(model, input_data=(x, t))
    print(out.shape)