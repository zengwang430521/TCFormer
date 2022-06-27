import sys
import os
dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, '../..'))

from tcformer_module.tcformer import tcformer, tcformer_large
from tcformer_module.mta_block import MTA
from mmpose.models.builder import BACKBONES, NECKS

BACKBONES.register_module(tcformer)
BACKBONES.register_module(tcformer_large)
NECKS.register_module(MTA)


# import torch
# backbone = tcformer()
# neck = MTA()
# x = torch.rand([2, 3, 224, 224])
# x = backbone(x)
# x = neck(x)
# print(x)