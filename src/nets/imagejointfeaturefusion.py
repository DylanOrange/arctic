import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ImageJointfeaturefusion(nn.Module):
    def __init__(self, in_dim, out_dim):#256,128
        super(ImageJointfeaturefusion, self).__init__()
        self.filters = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )
        self.pos_emb = nn.Sequential(
            nn.Conv1d(3, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, 1)
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(3*out_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, 1)
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

    def forward(self, img_feat, field_feat, joint_uv, joint_xyz):
        img_feat = F.grid_sample(img_feat, joint_uv.unsqueeze(1).detach()).squeeze(-2)#64,256,21
        img_feat = self.filters(img_feat)#64,128,21

        coord_feat = self.pos_emb(joint_xyz.permute(0, 2, 1).contiguous())#64,128,21

        fused_feat = self.fusion(torch.concat([img_feat, field_feat, coord_feat], dim=1))
        return fused_feat.permute(0,2,1)