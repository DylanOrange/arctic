import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# Define the Transformer Fusion Module with Cross-Attention and Single Fused Feature
class Transformer(nn.Module):
    def __init__(self, image_feature_dim, point_cloud_feature_dim, num_heads, hidden_dim):
        super(Transformer, self).__init__()

        # Linear projections for image and point cloud features
        self.image_projection = nn.Linear(image_feature_dim, hidden_dim)
        self.point_cloud_projection = nn.Linear(point_cloud_feature_dim, hidden_dim)

        # Multi-Head Self-Attention for Image Features
        # self.image_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first = True)

        # Multi-Head Self-Attention for Point Cloud Features
        self.point_cloud_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first = True)

        # Feed-Forward Networks
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, image_feature_dim),
            nn.ReLU(),
            nn.Linear(image_feature_dim, image_feature_dim),
            nn.ReLU()
        )
        
        # self.point_cloud_ffn = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image_features, point_cloud_features):
        #
        B, N, h, w = image_features.shape
        image_features = image_features.view(B,N,-1).permute(0,2,1)#64,49,2048
        point_cloud_features = point_cloud_features.permute(0,2,1)#64,195,1024

        # Project image and point cloud features to a common hidden dimension
        image_proj = self.image_projection(image_features)#64,49,1024
        point_cloud_proj = self.point_cloud_projection(point_cloud_features)#64,195,1024

        # Compute self-attended representations for image and point cloud features
        # image_self_attention_output, _ = self.image_attention(image_proj, image_proj, image_proj)
        # point_cloud_self_attention_output, _ = self.point_cloud_attention(point_cloud_proj, point_cloud_proj, point_cloud_proj)

        # Cross-Attention: Image Features attending to Point Cloud Features
        image_to_point_cloud_attention_output, _ = self.point_cloud_attention(
            image_proj, point_cloud_proj, point_cloud_proj
        )

        # Combine Cross-Attention Results
        combined_features = self.layer_norm(image_to_point_cloud_attention_output)#64,49,1024

        # Feed-Forward Network
        fused_feature = self.image_ffn(combined_features)

        fused_feature = fused_feature.permute(0,2,1).view(B,N,h,w)

        return fused_feature

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.changedim = changedim
        # self.currentdim = currentdim
        # self.depth = depth
        # if self.changedim:
        #     assert self.depth>0
        self.fc1 = nn.Linear(in_features, hidden_features)
        # nn.init.kaiming_normal_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.normal_(self.fc1.bias, std = 1e-6)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # nn.init.kaiming_normal_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.normal_(self.fc2.bias, std = 1e-6)

        self.drop = nn.Dropout(drop)
        # if self.changedim and self.currentdim <= self.depth//2:
        #     self.reduction = nn.Linear(out_features, out_features//2)
        # elif self.changedim and self.currentdim > self.depth//2:
        #     self.improve = nn.Linear(out_features, out_features*2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # if self.changedim and self.currentdim <= self.depth//2:
        #     x = self.reduction(x)
        # elif self.changedim and self.currentdim > self.depth//2:
        #     x = self.improve(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False,
                vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # nn.init.kaiming_normal_(self.qkv.weight)
        # torch.nn.init.xavier_uniform_(self.qkv.weight)
        # torch.nn.init.zeros_(self.qkv.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # nn.init.kaiming_normal_(self.proj.weight)
        # torch.nn.init.xavier_uniform_(self.proj.weight)
        # torch.nn.init.zeros_(self.proj.bias)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0.,
                attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x


class attn_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride=2, padding_mode='zeros'):
        super(attn_pooling, self).__init__()

        self.conv = nn.Conv1d(in_feature, out_feature, kernel_size=stride + 1,
                            padding=stride // 2, stride=stride,
                            padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        x = self.conv(x)
        return x


class STE(nn.Module):
    def __init__(self, num_joints=17, in_chans=32, out_dim=32, depth=4,
                num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, in_chans))
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            Block(dim=in_chans, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer)
            for i in range(depth)])

        self.spatial_norm = norm_layer(in_chans)

        self.head = nn.Sequential(
            nn.LayerNorm(in_chans),
            nn.Linear(in_chans, out_dim),
        )

    def forward(self, x):
        b, j, c = x.shape
        x += self.spatial_pos_embed
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            x = steblock(x)
            x = self.spatial_norm(x)

        x = self.head(x)
        x = x.view(b,-1, j)

        return x
