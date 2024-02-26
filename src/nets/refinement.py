import torch
import torch.nn as nn
import pytorch3d.transforms.rotation_conversions as rot_conv
import torch.nn.functional as F

from common.xdict import xdict
from src.nets.hmr_layer import HMRLayer
from src.nets.backbone.residual import Residual
from src.nets.transformer import STE
from src.nets.imagejointfeaturefusion import ImageJointfeaturefusion
from src.nets.pointnet_utils import normalize_point_cloud_torch
from src.nets.pointnet import PointNetfeat, PointNet2, ReconDecoder

class Refinement(nn.Module):
    def __init__(self, feat_dim, hand_joint_num, object_joint_num, hand_specs, obj_specs):
        super(Refinement, self).__init__()

        self.hand_joint_num = hand_joint_num
        self.object_joint_num = object_joint_num

        # self.fuse_img_joint_features_l = ImageJointfeaturefusion(256,128)
        # self.fuse_img_joint_features_r = ImageJointfeaturefusion(256,128)
        # self.fuse_img_joint_features_o = ImageJointfeaturefusion(256,128)

        # self.interaction_rl = STE(num_joints = (self.hand_joint_num * 2), in_chans=128, out_dim=64, depth=4)
        # self.interaction_or = STE(num_joints = (self.hand_joint_num + self.object_joint_num), in_chans=128, out_dim=64, depth=4)
        # self.interaction_ol = STE(num_joints = (self.hand_joint_num + self.object_joint_num), in_chans=128, out_dim=64, depth=4)
        self.interaction_o = STE(num_joints = self.object_joint_num, in_chans=512, out_dim=256, depth=4)

        self.refine_hmr_layer_l = HMRLayer(2048+512, 1024, hand_specs)
        self.refine_hmr_layer_r = HMRLayer(2048+512, 1024, hand_specs)
        self.refine_hmr_layer_o = HMRLayer(256*(self.object_joint_num), 1024, obj_specs)

        self.point_backbone_r = PointNet2(in_channel=6)
        self.point_backbone_l = PointNet2(in_channel=6)
        self.point_backbone_o = PointNetfeat(input_dim = 3, shallow_dim=128, mid_dim=128, out_dim=128)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

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
    
    def forward(self, img_feat, img_feat_full, field_feat_r, field_feat_l, field_feat_o, \
                mano_output_r, mano_output_l, arti_output, \
                hmr_output_r, hmr_output_l, hmr_output_o):
        
        root_r = mano_output_r["cam_t.r"][:,None,:]
        root_l = mano_output_l["cam_t.l"][:,None,:]

        vertex_r_3d = mano_output_r["v3d.cam.r"] - root_r
        vertex_r_3d, center_r, scale_r = normalize_point_cloud_torch(vertex_r_3d)
        vertex_l_3d = mano_output_l["v3d.cam.l"] - root_l        
        vertex_l_3d, center_l, scale_l = normalize_point_cloud_torch(vertex_l_3d)

        field_feat_r = torch.concat([vertex_r_3d, field_feat_r/scale_r], dim=2).permute(0,2,1)#16,6,778
        field_feat_l = torch.concat([vertex_l_3d, field_feat_l/scale_l], dim=2).permute(0,2,1)#16,6,778

        # field_feat_r = vertex_r_3d.permute(0,2,1)#16,6,778
        # field_feat_l = vertex_l_3d.permute(0,2,1)#16,6,778

        spatial_field_feat_r = self.point_backbone_r(field_feat_r)#16,384,21 16,1088,21
        spatial_field_feat_l = self.point_backbone_l(field_feat_l)#16,384,21

        img_feat_full = self.avgpool(img_feat_full)
        img_feat_full = img_feat_full.view(img_feat_full.size(0), -1)

        # img_feat_r = self.fuse_img_joint_features_r(img_feat, field_feat_r, mano_output_r["j2d.norm.r"], mano_output_r["j3d.cam.r"])#64,21,128
        # img_feat_l = self.fuse_img_joint_features_l(img_feat, field_feat_l, mano_output_l["j2d.norm.l"], mano_output_l["j3d.cam.l"])#64,21,128
        # img_feat_o = self.fuse_img_joint_features_o(img_feat, field_feat_o, arti_output["kp2d.norm"], arti_output["kp3d.cam"])#64,32,128

        field_feat_o = self.point_backbone_o(arti_output["kp3d.cam"].permute(0, 2, 1))[0]#16,384,32
        img_feat_o = F.grid_sample(img_feat, arti_output["kp2d.norm"].unsqueeze(1).detach()).squeeze(-2)#64,256,21
        img_feat_o = torch.concat([img_feat_o, field_feat_o], dim=1)
        img_feat_o = self.interaction_o(img_feat_o.permute(0,2,1))

        B,C,_ = img_feat_o.shape#64,21,128

        refine_dict_r = self.prepare_hand(hmr_output_r)
        refine_dict_l = self.prepare_hand(hmr_output_l)
        refine_dict_o = self.prepare_object(hmr_output_o)

        refine_mano_output_r = self.refine_hmr_layer_r(torch.concat([img_feat_full, spatial_field_feat_r],dim=1), refine_dict_r, n_iter = 3)
        refine_mano_output_l = self.refine_hmr_layer_l(torch.concat([img_feat_full, spatial_field_feat_l],dim=1), refine_dict_l, n_iter = 3)
        refine_mano_output_o = self.refine_hmr_layer_o(img_feat_o.reshape(B,C*(self.object_joint_num)), refine_dict_o, n_iter = 3)

        refine_mano_output_r["pose"] = rot_conv.rotation_6d_to_matrix(
            refine_mano_output_r["pose_6d"].reshape(-1, 6)).view(B, 16, 3, 3)
        refine_mano_output_r = refine_mano_output_r.replace_keys("/", ".")

        refine_mano_output_l["pose"] = rot_conv.rotation_6d_to_matrix(
            refine_mano_output_l["pose_6d"].reshape(-1, 6)).view(B, 16, 3, 3)
        refine_mano_output_l = refine_mano_output_l.replace_keys("/", ".")

        refine_mano_output_o = refine_mano_output_o.replace_keys("/", ".")

        return refine_mano_output_r, refine_mano_output_l, refine_mano_output_o