import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix

from src.nets.transformer_decoder import TransformerDecoder

class MANOTransformerDecoderHead(nn.Module):
    """ Cross-attention based MANO Transformer decoder
    """

    def __init__(self):
        super().__init__()
        self.joint_rep_type = '6d'
        self.joint_rep_dim = 6
        npose = 96
        self.npose = npose
        self.input_is_mean_shape = False
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = (transformer_args | dict(
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm='layer',
            context_dim=1280,
        ))
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        mean_params = np.load('docs/mano_mean_params.npz')
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):

        batch_size = x.shape[0]#2,1280,16,12
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')#2,192,1280

        pred_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        pred_betas = self.init_betas.expand(batch_size, -1)
        pred_cam = self.init_cam.expand(batch_size, -1)

        # Input token to transformer is zero token
        token = torch.zeros(batch_size, 1, 1).to(x.device)
        # Pass through transformer
        # first input, token=2,1,1, x=2,192,1280, out = 2,1,1024
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1) # (B, C)

        # Readout from token_out
        pred_hand_pose = self.decpose(token_out) + pred_hand_pose
        pred_betas = self.decshape(token_out) + pred_betas
        pred_cam = self.deccam(token_out) + pred_cam

        # Convert self.joint_rep_type -> rotmat
        pred_hand_pose = rotation_6d_to_matrix(pred_hand_pose.reshape(batch_size*16, 6)).view(batch_size, 16, 3, 3)
        return pred_hand_pose, pred_betas, pred_cam
