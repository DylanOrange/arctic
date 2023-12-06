import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Enhancement(nn.Module):
    def __init__(self, hand_joint_num, object_joint_num, distance=1):
        super(Enhancement, self).__init__()

        self.feature_size = 7
        self.hand_joint_num = hand_joint_num
        self.object_joint_num = object_joint_num

        x = (torch.arange(self.feature_size) + 0.5)
        y = (torch.arange(self.feature_size) + 0.5)

        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.img_gird = torch.stack((grid_y, grid_x), dim=-1).reshape([self.feature_size ** 2, 2]).contiguous()

        self.distance = distance

        self.proj_feat_emb = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1)
        )

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def bone_proj(self, joint_uv, joint_feat):

        device = joint_uv.device
        B, J, C = joint_feat.size()#B,21,128
        S = self.feature_size#7
        joint_uv = (joint_uv+1)/2*S#unnormalize 64,21,2

        joint_uv = joint_uv.reshape([B, 1, J, 2]).repeat(1, S ** 2, 1, 1).reshape([-1, 2])#B,1,21,2->B,49,21,2->(65856,2)
        joint_feat = joint_feat.reshape([B, 1, J, -1])#B,1,21,128

        img_gird = self.img_gird.reshape([1, S ** 2, 1, 2]).repeat(B, 1, J, 1).reshape(-1, 2).to(device)#1,49,1,2->B,49,20,2->(65856,2)

        distance = F.pairwise_distance(img_gird, joint_uv, p=2)
        distance = torch.clamp(distance, min=0, max=self.distance)
    
        weight = 1 - (distance / self.distance)
        weight = weight.reshape([B, -1, J, 1])

        img_feat = joint_feat * weight#64,49,21,128
        img_feat = img_feat.reshape([B, S, S, J*C]).permute(0, 3, 1, 2)#B,7,7,21*128->B,21*128,7,7

        return img_feat
    
    def forward(self, mano_output_r, mano_output_l, arti_output, img_feat_r, img_feat_l, img_feat_o):
           
        img_feat_r = self.proj_feat_emb(img_feat_r).permute(0,2,1)#64,128,21->64,21,128
        img_feat_l = self.proj_feat_emb(img_feat_l).permute(0,2,1)#64,128,21->64,21,128
        img_feat_o = self.proj_feat_emb(img_feat_o).permute(0,2,1)#64,128,32->64,32,128

        img_feat_r = self.bone_proj(mano_output_r["j2d.norm.r"], img_feat_r)#64,21,2
        img_feat_l = self.bone_proj(mano_output_l["j2d.norm.l"], img_feat_l)#64,21,2
        img_feat_o = self.bone_proj(arti_output["kp2d.norm"], img_feat_o)#64,32,2

        img_feat = torch.cat((img_feat_r, img_feat_l, img_feat_o), dim=1)

        return img_feat#64,256,7,7