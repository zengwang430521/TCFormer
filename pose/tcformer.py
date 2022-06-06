import sys
# sys.path.insert(0, '..')
import os
dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, '..'))

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