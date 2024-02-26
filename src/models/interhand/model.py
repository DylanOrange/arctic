import torch
import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.models.arctic_sf.model import ArcticSF
from src.models.field_sf.model import FieldSF
from src.nets.backbone.utils import get_backbone_info
import src.callbacks.process.process_generic as generic


class InterHand(nn.Module):
    def __init__(self, backbone, args):
        super(InterHand, self).__init__()

        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as img_backbone
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as img_backbone
        elif backbone == "ViT":
            from src.nets.backbone.ViT import vit_base_patch16_224 as img_backbone
        elif backbone == "ViT-L":
            from src.nets.backbone.ViT import vit_large_patch16_224 as img_backbone
        elif backbone =='ViT-H':
            from src.nets.backbone.vit import vit as img_backbone
        else:
            assert False
        self.backbone = img_backbone(pretrained=True)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]

        self.arctic_model = ArcticSF(backbone, args.focal_length, args.img_res, args)

        self.field_model = FieldSF(backbone, args.focal_length, args.img_res)

        self.args = args

        self.feat_dim = feat_dim
        self.object_kpnum = 32
        self.hand_kpnum = 21
        self.field_value = 0.1
        self.num_patch = 14

    def forward(self, inputs, targets, meta_info):
        #field initialization to 0.1
        # B = inputs["img"].shape[0]
        # device = inputs["img"].device

        #feature extraction
        images = inputs["img"]#64,3,224,224
        features = self.backbone(images)
        # features = self.backbone(images)[:,1:]#64,2048,7,7 for resnet, 64,196,768 for ViT, 16,1280,14,14 for vit
        # features = features.permute(0,2,1).reshape(-1,self.feat_dim,self.num_patch,self.num_patch).contiguous()#64,196,768->64,768,196->64,768,14,14

        updated_field = self.field_model(features, meta_info)

            #hand/object pose prediction
        output = self.arctic_model(features, meta_info, updated_field)

            #recalculate field
        # output = generic.prepare_kp_interfield(output, float("inf"), True)
        
        #merge output predicted field key
        output.update(updated_field)

        return output
