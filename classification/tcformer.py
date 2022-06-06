import sys
sys.path.insert(0, '..')
import torch.nn as nn
from timm.models.registry import register_model
from tcformer_module.tcformer import TCFormer as _TCFormer
from functools import partial


class TCFormer(_TCFormer):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        # classification head
        self.head = nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(self._init_weights)

    def forward(self, x):
        x = self.forward_features(x)
        x = x[-1]['x'].mean(dim=1)
        x = self.head(x)
        return x

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


@register_model
def tcformer_light(pretrained=False, **kwargs):
    model = TCFormer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model


@register_model
def tcformer(pretrained=False, **kwargs):
    model = TCFormer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model


@register_model
def tcformer_large(pretrained=False, **kwargs):
    model = TCFormer(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


