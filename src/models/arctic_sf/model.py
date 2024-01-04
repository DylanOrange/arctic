import torch
import torch.nn as nn
import numpy as np
import os

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
from src.nets.mano_decoder import MANOTransformerDecoderHead


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
        elif backbone == "ViT-L":
            from src.nets.backbone.ViT import vit_large_patch16_224 as img_backbone
        elif backbone =='ViT-H':
            from src.nets.backbone.vit import vit as img_backbone
        else:
            assert False
        self.backbone = img_backbone(pretrained=True)
        # self.mano_head = MANOTransformerDecoderHead()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

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
        # self.head_o = ObjectHMR(feat_dim, n_iter=3)

        # self.img_encoder = Residual(feat_dim, 256)
        self.regressor = Regressor(focal_length=focal_length, img_res=img_res)
        # self.refinement1 = Refinement(feat_dim, self.hand_joint_num, self.object_joint_num, hand_specs, obj_specs)
        # self.refinement2 = Refinement(256, self.hand_joint_num, self.object_joint_num, hand_specs, obj_specs)
        # self.enhancement = Enhancement(self.hand_joint_num, self.object_joint_num)
        # self.enhance_layer = Residual(64 * (self.hand_joint_num * 2 + self.object_joint_num)+256, 256)

        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length
    
    # def forward(self, features, meta_info, x_t, time_emb, field):
    #     # images = inputs["img"]#64,3,224,224

    #     field_r = field["dist.ro.kp"][:,None,:]#64,1,21
    #     field_l = field["dist.lo.kp"][:,None,:]#64,1,21
    #     field_o = torch.stack([field["dist.or.kp"], field["dist.ol.kp"]], dim=1)#64,2,32

    #     field_feat_r= self.point_backbone_h(field_r)[0]#64,128,21
    #     field_feat_l= self.point_backbone_h(field_l)[0]#64,128,21
    #     field_feat_o= self.point_backbone_o(field_o)[0]#64,128,32

    #     #backbone
    #     # features = self.backbone(images)#64,2048,7,7 for resnet, 64,197,768 for ViT

    #     #recover the spatial dimension
    #     # features = features.permute(0,2,1).reshape(-1,self.feat_dim,14,14).contiguous()#64,196,768->64,768,196->64,768,14,14

    #     feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)#64,768

    #     hmr_output_r = self.head_r(features, x_t[:,:109], time_emb)#64,2048,7,7 -> 64,2048
    #     hmr_output_l = self.head_l(features, x_t[:,109:218], time_emb)
    #     hmr_output_o = self.head_o(features, x_t[:,-7:], time_emb)

    #     #initial regressor
    #     mano_output_r, mano_output_l, arti_output = self.regressor(hmr_output_r, hmr_output_l, hmr_output_o, meta_info)

    #     # second stage, refinement
    #     features = self.img_encoder(features)

    #     refine_mano_output_r, refine_mano_output_l, refine_mano_output_o, img_feat_r, img_feat_l, img_feat_o, img_feat_fusion\
    #           = self.refinement1(features, field_feat_r, field_feat_l, field_feat_o, \
    #                             mano_output_r, mano_output_l, arti_output, \
    #                                 hmr_output_r, hmr_output_l, hmr_output_o)  
    #     #second stage, regressor
    #     mano_output_r, mano_output_l, arti_output = self.regressor(refine_mano_output_r, \
    #                                                                refine_mano_output_l, refine_mano_output_o, meta_info)
        
    #     # #enhancement
    #     # # img_feat = self.enhancement(mano_output_r, mano_output_l, arti_output, img_feat_r, img_feat_l, img_feat_o)
    #     # # img_feat = torch.cat((img_feat_fusion, img_feat), dim=1)

    #     # #third stage, refinement
    #     # refine_mano_output_r, refine_mano_output_l, refine_mano_output_o, img_feat_r, img_feat_l, img_feat_o, img_feat_fusion\
    #     #       = self.refinement2(img_feat_fusion, field_feat_r, field_feat_l, field_feat_o, \
    #     #                         mano_output_r, mano_output_l, arti_output, \
    #     #                             hmr_output_r, hmr_output_l, hmr_output_o)
    #     # #third stage, regressor
    #     # mano_output_r, mano_output_l, arti_output = self.regressor(refine_mano_output_r, \
    #     #                                                            refine_mano_output_l, refine_mano_output_o, meta_info)

    #     #add init camera parameters
    #     root_r_init = hmr_output_r["cam_t.wp.init"]
    #     root_l_init = hmr_output_l["cam_t.wp.init"]
    #     root_o_init = hmr_output_o["cam_t.wp.init"]
    #     mano_output_r["cam_t.wp.init.r"] = root_r_init
    #     mano_output_l["cam_t.wp.init.l"] = root_l_init
    #     arti_output["cam_t.wp.init"] = root_o_init

    #     mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
    #     mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
    #     arti_output = ld_utils.prefix_dict(arti_output, "object.")
    #     output = xdict()
    #     output.merge(mano_output_r)
    #     output.merge(mano_output_l)
    #     output.merge(arti_output)
    #     output["feat_vec"] = feat_vec.cpu().detach()
    #     return output
    
    def forward(self, inputs, meta_info):

        B,N,dim,h,w = inputs["kp_img"].shape#16,2,3,256,256
        images = inputs["kp_img"].reshape(B*N,dim,h,w)#32,3,224,224

        #backbone
        features = self.backbone(images[:,:,:,32:-32])#64,2048,7,7 for resnet, 64,197,768 for ViT, 32,1280,16,12
        features = features.reshape(B,N,1280,16,12)
        feature_l = features[:,0,:,:,:]#16,1280,16,12
        feature_r = features[:,1,:,:,:]#16,1280,16,12
        
        # save_feat_l = feature_l.detach().cpu().numpy()
        # save_feat_r = feature_r.detach().cpu().numpy()

        # for i in range(B):
        #     split_dict = meta_info['save_img_name'][i].split('/')
        #     dict_path = '/storage/user/lud/dataset/arctic/data/arctic_data/data/saved_feat/'+'/'.join(split_dict[:3])
        #     if not os.path.exists(dict_path):
        #         os.makedirs(dict_path)
        #     file_l = os.path.join(dict_path, split_dict[-1]+'_l.npy')
        #     file_r = os.path.join(dict_path, split_dict[-1]+'_r.npy')
        #     np.save(file_l, save_feat_l[i])
        #     np.save(file_r, save_feat_r[i])

        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)#64,768

        hmr_output_r = self.head_r(feature_r)#64,2048,7,7 -> 64,2048
        hmr_output_l = self.head_l(feature_l)

        #initial regressor
        mano_output_r, mano_output_l = self.regressor(hmr_output_r, hmr_output_l, meta_info)

        #add init camera parameters
        root_r_init = hmr_output_r["cam_t.wp.init"]
        root_l_init = hmr_output_l["cam_t.wp.init"]
        # root_o_init = hmr_output_o["cam_t.wp.init"]
        mano_output_r["cam_t.wp.init.r"] = root_r_init
        mano_output_l["cam_t.wp.init.l"] = root_l_init
        # arti_output["cam_t.wp.init"] = root_o_init

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        # arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        # output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach()
        return output
    
    # def forward(self, inputs, meta_info):

    #     B,N,dim,h,w = inputs["kp_img"].shape#16,2,3,256,256
    #     images = inputs["kp_img"].reshape(B*N,dim,h,w)#32,3,224,224

    #     #backbone
    #     features = self.backbone(images[:,:,:,32:-32])#64,2048,7,7 for resnet, 64,197,768 for ViT, 32,1280,16,12
    #     # features = features.reshape(B,N,1280,16,12)
    #     # feature_l = features[:,0,:,:,:]#16,1280,16,12
    #     # feature_r = features[:,1,:,:,:]#16,1280,16,12

    #     # save_feat_l = feature_l.detach().cpu().numpy()
    #     # save_feat_r = feature_r.detach().cpu().numpy()

    #     # for i in range(B):
    #     #     split_dict = meta_info['save_img_name'][i].split('/')
    #     #     dict_path = '../original_packages/arctic/data/arctic_data/data/saved_feat/'+'/'.join(split_dict[:3])
    #     #     if not os.path.exists(dict_path):
    #     #         os.makedirs(dict_path)
    #     #     file_l = os.path.join(dict_path, split_dict[-1]+'_l.npy')
    #     #     file_r = os.path.join(dict_path, split_dict[-1]+'_r.npy')
    #     #     np.save(file_l, save_feat_l[i])
    #     #     np.save(file_r, save_feat_r[i])

    #     pose, beta, cam_t = self.mano_head(features)#32,16,3,3, 32,10, 32,3
    #     bihand_pose = pose.reshape(B,N,16,3,3)
    #     bihand_beta = beta.reshape(B,N,10)
    #     bihand_camt = cam_t.reshape(B,N,3)

    #     hmr_output_l = {}
    #     hmr_output_r = {}

    #     hmr_output_l["pose"] = bihand_pose[:,0,:,:,:]
    #     hmr_output_l["shape"] = bihand_beta[:,0,:]
    #     hmr_output_l["cam_t.wp"] = bihand_camt[:,0,:]

    #     hmr_output_r["pose"] = bihand_pose[:,1,:,:,:]
    #     hmr_output_r["shape"] = bihand_beta[:,1,:]
    #     hmr_output_r["cam_t.wp"] = bihand_camt[:,1,:]

    #     #initial regressor
    #     mano_output_r, mano_output_l = self.regressor(hmr_output_r, hmr_output_l, meta_info)

    #     mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
    #     mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
    #     # arti_output = ld_utils.prefix_dict(arti_output, "object.")
    #     output = xdict()
    #     output.merge(mano_output_r)
    #     output.merge(mano_output_l)
    #     # output.merge(arti_output)
    #     return output
