import torch
import torch.nn as nn
from functools import partial
import math
from .tcformer_layers import Block, TCBlock, OverlapPatchEmbed, CTM
from .tcformer_utils import (
    load_checkpoint, get_root_logger, token2map,
    token2map_flops, map2token_flops, cluster_and_merge_flops,
    downup_flops, sra_flops)
from .transformer_utils import trunc_normal_


class TCFormer(nn.Module):
    def __init__(
            self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            num_stages=4, pretrained=None,
            k=5, sample_ratios=[0.25, 0.25, 0.25],
            return_map=False,
            **kwargs
    ):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # In stage 1, use the standard transformer blocks
        for i in range(1):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # In stage 2~4, use TCBlock for dynamic tokens
        for i in range(1, num_stages):
            ctm = CTM(sample_ratios[i-1], embed_dims[i-1], embed_dims[i], k)

            block = nn.ModuleList([TCBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"ctm{i}", ctm)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def forward_features(self, x):
        outs = []

        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)

        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        # stage 2~4
        for i in range(1, self.num_stages):
            ctm = getattr(self, f"ctm{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            token_dict = ctm(token_dict)  # down sample
            for j, blk in enumerate(block):
                token_dict = blk(token_dict)

            token_dict['x'] = norm(token_dict['x'])
            outs.append(token_dict)

        if self.return_map:
            outs = [token2map(token_dict) for token_dict in outs]

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class tcformer_light(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class tcformer(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            nh_list=[1, 1, 1], nw_list=[1, 1, 1],
            **kwargs)


class tcformer_large(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


