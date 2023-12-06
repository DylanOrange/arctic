import torch
import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.obj_heads.obj_hmr import ObjectHMR
from src.nets.pointnet import PointNetfeat
from src.nets.regressor import Regressor
from src.nets.refinement import Refinement
from src.nets.enhancement import Enhancement
from src.nets.backbone.residual import Residual


class ArcticSF(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(ArcticSF, self).__init__()
        self.args = args
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as img_backbone
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as img_backbone
        elif backbone == "ViT":
            from src.nets.backbone.ViT import vit_base_patch16_224 as img_backbone
            self.spatial_h = 14 
        else:
            assert False
        self.backbone = img_backbone(pretrained=True)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]

        hand_specs = {"pose_6d": 6 * 16, "cam_t/wp": 3, "shape": 10}
        obj_specs = {"rot": 3, "cam_t/wp": 3, "radian": 1}

        pt_shallow_dim = 64
        pt_mid_dim = 64
        pt_out_dim = 64
        
        self.hand_joint_num = 21
        self.object_joint_num = 32

        # self.point_backbone_h = PointNetfeat(
        #     input_dim=1,
        #     shallow_dim=pt_shallow_dim,
        #     mid_dim=pt_mid_dim,
        #     out_dim=pt_out_dim,
        # )

        # self.point_backbone_o = PointNetfeat(
        #     input_dim=2,
        #     shallow_dim=pt_shallow_dim,
        #     mid_dim=pt_mid_dim,
        #     out_dim=pt_out_dim,
        # )

        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)
        self.head_o = ObjectHMR(feat_dim, n_iter=3)

        # self.img_encoder = Residual(feat_dim, 256)
        self.regressor = Regressor(focal_length=focal_length, img_res=img_res)
        # self.refinement1 = Refinement(feat_dim, self.hand_joint_num, self.object_joint_num, hand_specs, obj_specs)
        # self.refinement2 = Refinement(256, self.hand_joint_num, self.object_joint_num, hand_specs, obj_specs)
        # self.enhancement = Enhancement(self.hand_joint_num, self.object_joint_num)
        # self.enhance_layer = Residual(64 * (self.hand_joint_num * 2 + self.object_joint_num)+256, 256)

        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length
    
    def forward(self, inputs, meta_info):
        images = inputs["img"]#64,3,224,224

        # field_r = meta_info["dist.ro"][:,None,:]#64,1,21
        # field_l = meta_info["dist.lo"][:,None,:]#64,1,21
        # field_o = torch.stack([meta_info["dist.or"], meta_info["dist.ol"]], dim=1)#64,2,32

        # field_feat_r= self.point_backbone_h(field_r)[0]#64,128,21
        # field_feat_l= self.point_backbone_h(field_l)[0]#64,128,21
        # field_feat_o= self.point_backbone_o(field_o)[0]#64,128,32

        #backbone
        features = self.backbone(images)#64,2048,7,7 for resnet, 64,197,768 for ViT

        #recover the spatial dimension
        # features = features[:,1:,:].view(-1,self.spatial_h,self.spatial_h,features.shape[-1])#64,14,14,768
        # features = features.permute(0,3,1,2)#64,768,14,14

        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)#64,768

        hmr_output_r = self.head_r(features)#64,2048,7,7 -> 64,2048
        hmr_output_l = self.head_l(features)
        hmr_output_o = self.head_o(features)

        #initial regressor
        mano_output_r, mano_output_l, arti_output = self.regressor(hmr_output_r, hmr_output_l, hmr_output_o, meta_info)

        # second stage, refinement
        # features = self.img_encoder(features)

        # refine_mano_output_r, refine_mano_output_l, refine_mano_output_o, img_feat_r, img_feat_l, img_feat_o, img_feat_fusion\
        #       = self.refinement1(features, field_feat_r, field_feat_l, field_feat_o, \
        #                         mano_output_r, mano_output_l, arti_output, \
        #                             hmr_output_r, hmr_output_l, hmr_output_o)  
        # #second stage, regressor
        # mano_output_r, mano_output_l, arti_output = self.regressor(refine_mano_output_r, \
        #                                                            refine_mano_output_l, refine_mano_output_o, meta_info)
        
        # #enhancement
        # # img_feat = self.enhancement(mano_output_r, mano_output_l, arti_output, img_feat_r, img_feat_l, img_feat_o)
        # # img_feat = torch.cat((img_feat_fusion, img_feat), dim=1)

        # #third stage, refinement
        # refine_mano_output_r, refine_mano_output_l, refine_mano_output_o, img_feat_r, img_feat_l, img_feat_o, img_feat_fusion\
        #       = self.refinement2(img_feat_fusion, field_feat_r, field_feat_l, field_feat_o, \
        #                         mano_output_r, mano_output_l, arti_output, \
        #                             hmr_output_r, hmr_output_l, hmr_output_o)
        # #third stage, regressor
        # mano_output_r, mano_output_l, arti_output = self.regressor(refine_mano_output_r, \
        #                                                            refine_mano_output_l, refine_mano_output_o, meta_info)

        #add init camera parameters
        root_r_init = hmr_output_r["cam_t.wp.init"]
        root_l_init = hmr_output_l["cam_t.wp.init"]
        root_o_init = hmr_output_o["cam_t.wp.init"]
        mano_output_r["cam_t.wp.init.r"] = root_r_init
        mano_output_l["cam_t.wp.init.l"] = root_l_init
        arti_output["cam_t.wp.init"] = root_o_init

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach()
        return output
