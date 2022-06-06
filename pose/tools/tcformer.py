import sys
sys.path.insert(0, '../..')
from tcformer_module.tcformer import tcformer
from tcformer_module.mta_block import MTA
from mmpose.models.builder import BACKBONES, NECKS

BACKBONES.register_module(tcformer)
NECKS.register_module(MTA)


# import torch
# backbone = tcformer()
# neck = MTA()
# x = torch.rand([2, 3, 224, 224])
# x = backbone(x)
# x = neck(x)
# print(x)