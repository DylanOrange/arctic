import torch
import torch.nn as nn
import pytorch3d.transforms.rotation_conversions as rot_conv

from common.xdict import xdict
from src.nets.hmr_layer import HMRLayer
from src.nets.backbone.residual import Residual
from src.nets.transformer import STE
from src.nets.imagejointfeaturefusion import ImageJointfeaturefusion

class Refinement(nn.Module):
    def __init__(self, feat_dim, hand_joint_num, object_joint_num, hand_specs, obj_specs):
        super(Refinement, self).__init__()

        self.hand_joint_num = hand_joint_num
        self.object_joint_num = object_joint_num

        self.fuse_img_joint_features_l = ImageJointfeaturefusion(256,128)
        self.fuse_img_joint_features_r = ImageJointfeaturefusion(256,128)
        self.fuse_img_joint_features_o = ImageJointfeaturefusion(256,128)

        # self.interaction_rl = STE(num_joints = (self.hand_joint_num * 2), in_chans=128, out_dim=64, depth=4)
        # self.interaction_or = STE(num_joints = (self.hand_joint_num + self.object_joint_num), in_chans=128, out_dim=64, depth=4)
        # self.interaction_ol = STE(num_joints = (self.hand_joint_num + self.object_joint_num), in_chans=128, out_dim=64, depth=4)

        self.refine_hmr_layer_l = HMRLayer(128*(self.hand_joint_num), 1024, hand_specs)
        self.refine_hmr_layer_r = HMRLayer(128*(self.hand_joint_num), 1024, hand_specs)
        self.refine_hmr_layer_o = HMRLayer(128*(self.object_joint_num), 1024, obj_specs)

        # self.feat_emb_l = nn.Sequential(
        #     nn.Conv1d(128, 64, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, 1)
        # )
        # self.feat_emb_r = nn.Sequential(
        #     nn.Conv1d(128, 64, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, 1)
        # )
        # self.feat_emb_o = nn.Sequential(
        #     nn.Conv1d(128, 64, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, 1)
        # )
    def prepare_hand(self,dict):

        out = {}
        out["pose_6d"] = dict["pose_6d"]#64,16*6
        out["shape"] = dict["shape"]#64,10
        out["cam_t/wp"] = dict["cam_t.wp"]#64,3
        out = xdict(out).to(dict["cam_t.wp"].device)

        return out
    
    def prepare_object(self,dict):

        out = {}
        out["rot"] = dict["rot"]#64,16*6
        out["radian"] = dict["radian"]#64,10
        out["cam_t/wp"] = dict["cam_t.wp"]#64,3
        out = xdict(out).to(dict["cam_t.wp"].device)

        return out
    
    def forward(self, img_feat, field_feat_r, field_feat_l, field_feat_o, \
                mano_output_r, mano_output_l, arti_output, \
                hmr_output_r, hmr_output_l, hmr_output_o):

        img_feat_r = self.fuse_img_joint_features_r(img_feat, field_feat_r, mano_output_r["j2d.norm.r"], mano_output_r["j3d.cam.r"])#64,21,128
        img_feat_l = self.fuse_img_joint_features_l(img_feat, field_feat_l, mano_output_l["j2d.norm.l"], mano_output_l["j3d.cam.l"])#64,21,128
        img_feat_o = self.fuse_img_joint_features_o(img_feat, field_feat_o, arti_output["kp2d.norm"], arti_output["kp3d.cam"])#64,32,128

        # img_feat_rl = self.interaction_rl(torch.cat([img_feat_r, img_feat_l], dim=1))
        # img_feat_or = self.interaction_or(torch.cat([img_feat_o, img_feat_r], dim=1))
        # img_feat_ol = self.interaction_ol(torch.cat([img_feat_o, img_feat_l], dim=1))

        # img_feat_rl_r, img_feat_rl_l = torch.split(img_feat_rl, [self.hand_joint_num, self.hand_joint_num], dim=2)
        # img_feat_or_o, img_feat_or_r = torch.split(img_feat_or, [self.object_joint_num, self.hand_joint_num], dim=2)
        # img_feat_ol_o, img_feat_ol_l = torch.split(img_feat_ol, [self.object_joint_num, self.hand_joint_num], dim=2)

        # img_feat_r = self.feat_emb_r(torch.cat([img_feat_rl_r, img_feat_or_r], dim=1))
        # img_feat_l = self.feat_emb_l(torch.cat([img_feat_rl_l, img_feat_ol_l], dim=1))
        # img_feat_o = self.feat_emb_o(torch.cat([img_feat_or_o, img_feat_ol_o], dim=1))

        B,C,_ = img_feat_l.shape#64,128,21

        refine_dict_r = self.prepare_hand(hmr_output_r)
        refine_dict_l = self.prepare_hand(hmr_output_l)
        refine_dict_o = self.prepare_object(hmr_output_o)

        refine_mano_output_r = self.refine_hmr_layer_r(img_feat_r.reshape(B,C*(self.hand_joint_num)), refine_dict_r, n_iter = 3)
        refine_mano_output_l = self.refine_hmr_layer_l(img_feat_l.reshape(B,C*(self.hand_joint_num)), refine_dict_l, n_iter = 3)
        refine_mano_output_o = self.refine_hmr_layer_o(img_feat_o.reshape(B,C*(self.object_joint_num)), refine_dict_o, n_iter = 3)

        refine_mano_output_r["pose"] = rot_conv.rotation_6d_to_matrix(
            refine_mano_output_r["pose_6d"].reshape(-1, 6)).view(B, 16, 3, 3)
        refine_mano_output_r = refine_mano_output_r.replace_keys("/", ".")

        refine_mano_output_l["pose"] = rot_conv.rotation_6d_to_matrix(
            refine_mano_output_l["pose_6d"].reshape(-1, 6)).view(B, 16, 3, 3)
        refine_mano_output_l = refine_mano_output_l.replace_keys("/", ".")

        refine_mano_output_o = refine_mano_output_o.replace_keys("/", ".")

        return refine_mano_output_r, refine_mano_output_l, refine_mano_output_o, img_feat_r, img_feat_l, img_feat_o, img_feat