import torch
import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR
from src.nets.pointnet import PointNetfeat
from src.nets.transformer import Transformer

class ArcticSF(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(ArcticSF, self).__init__()
        self.args = args
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as resnet
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as resnet
        else:
            assert False
        self.backbone = resnet(pretrained=True)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]

        pt_shallow_dim = 512
        pt_mid_dim = 512
        pt_out_dim = 512
        num_heads = 4
        hidden_dim = 1024

        self.transformer = Transformer(image_feature_dim=feat_dim, point_cloud_feature_dim=pt_out_dim+pt_mid_dim, 
                                       num_heads=num_heads, hidden_dim=hidden_dim)
        
        self.point_backbone_h = PointNetfeat(
            input_dim=1,
            shallow_dim=pt_shallow_dim,
            mid_dim=pt_mid_dim,
            out_dim=pt_out_dim,
        )

        self.point_backbone_o = PointNetfeat(
            input_dim=2,
            shallow_dim=pt_shallow_dim,
            mid_dim=pt_mid_dim,
            out_dim=pt_out_dim,
        )

        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)

        self.head_o = ObjectHMR(feat_dim, n_iter=3)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mano_l = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)
        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]

        #fetch points coordinate
        # points_r = meta_info["v0.r.full"].permute(0, 2, 1)[:, :, 21:]
        # points_l = meta_info["v0.l.full"].permute(0, 2, 1)[:, :, 21:]
        # points_or = meta_info["v0.o.full"].permute(0, 2, 1)
        # points_ol = meta_info["v0.o.full"].permute(0, 2, 1)

        field_r = meta_info["dist.ro"][:,None,:]#64,1,195
        field_l = meta_info["dist.lo"][:,None,:]#64,1,195
        field_o = torch.stack([meta_info["dist.or"], meta_info["dist.ol"]], dim=1)#64,2,600

        features_r= self.point_backbone_h(field_r)[0]#64,1024,195
        features_l= self.point_backbone_h(field_l)[0]#64,1024,195
        features_o= self.point_backbone_o(field_o)[0]#64,1024,600

        #backbone
        features = self.backbone(images)#64,2048,7,7
        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)

        #fuse image and pointcloud features
        features_r = self.transformer(features, features_r)
        features_l = self.transformer(features, features_l)
        features_o = self.transformer(features, features_o)

        hmr_output_r = self.head_r(features_r)#64,2048,7,7 -> 64,2048
        hmr_output_l = self.head_l(features_l)
        hmr_output_o = self.head_o(features_o)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]
        root_o = hmr_output_o["cam_t.wp"]
        
        #mano head
        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l["pose"],
            shape=hmr_output_l["shape"],
            K=K,
            cam=root_l,
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=hmr_output_o["rot"],
            angle=hmr_output_o["radian"],
            query_names=query_names,
            cam=root_o,
            K=K,
        )

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
