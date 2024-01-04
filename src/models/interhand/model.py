import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.models.arctic_sf.model import ArcticSF
from src.models.field_sf.model import FieldSF
import src.callbacks.process.process_generic as generic
from src.nets.backbone.utils import get_backbone_info
from src.utils.sampler import UniformSampler, get_named_beta_schedule, extract_into_tensor, space_timesteps, PositionalEncoding, TimestepEmbedder
from src.callbacks.loss.loss_field import diffusion_dist_loss_kp

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
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # self.backbone.pos_embedding.requires_grad = True
            
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        
        self.arctic_model = ArcticSF(backbone, args.focal_length, args.img_res, args)

        self.field_model = FieldSF(backbone, args.focal_length, args.img_res)
        
        self.args = args
        
        self.feat_dim = feat_dim
        self.object_kpnum = 32
        self.hand_kpnum = 21
        self.field_value = 0.1
        self.num_patch = 14
        self.para_dim = 2 * (6*16 + 3 + 10) + (3 + 3 + 1)
        self.scale = 10000.0
        
        #diffusion parameters
        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'
        
        self.sampler = UniformSampler(num_timesteps)
        self.sequence_pos_encoder = PositionalEncoding(512, 0)
        self.embed_timestep = TimestepEmbedder(512, self.sequence_pos_encoder)

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(beta_scheduler, num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
    def field_initialization(self, batch, device):
        
        field = xdict()
        field["dist.ro.kp"] = self.field_value * torch.ones((batch,self.hand_kpnum), device=device)
        field["dist.lo.kp"] = self.field_value * torch.ones((batch,self.hand_kpnum), device=device)
        field["dist.or.kp"] = self.field_value * torch.ones((batch,self.object_kpnum), device=device)
        field["dist.ol.kp"] = self.field_value * torch.ones((batch,self.object_kpnum), device=device)

        return field
    
    def generate_noise(self, features):
        
        noise = torch.randn((features.shape[0], self.para_dim), device=features.device, dtype=features.dtype)
        return noise
    
    def input_process(self, targets):
        
        gt_pose_r = matrix_to_rotation_6d(axis_angle_to_matrix(targets["mano.pose.r"].reshape(-1, 3))).reshape(-1,16*6)#axis->matrix->6d #64,96
        gt_shape_r = targets["mano.beta.r"]
        gt_cam_r = targets["mano.cam_t.wp.r"]
        gt_r = torch.cat([gt_pose_r, gt_shape_r, gt_cam_r], dim=-1)

        gt_pose_l = matrix_to_rotation_6d(axis_angle_to_matrix(targets["mano.pose.l"].reshape(-1, 3))).reshape(-1,16*6)
        gt_shape_l = targets["mano.beta.l"]
        gt_cam_l = targets["mano.cam_t.wp.l"]
        gt_l = torch.cat([gt_pose_l, gt_shape_l, gt_cam_l], dim=-1)

        gt_rot_o = targets["object.rot"].squeeze(1)
        gt_radian_o = targets["object.radian"].unsqueeze(-1)
        gt_cam_o = targets["object.cam_t.wp"]
        gt_o = torch.cat([gt_rot_o, gt_radian_o, gt_cam_o], dim=-1)
        
        x_0 = torch.cat([gt_r, gt_l, gt_o], dim=-1)
        
        return x_0
    
    def output_process(self, targets):
        
        gt_pose_r = matrix_to_rotation_6d(targets["mano.pose.r"].reshape(-1,3,3)).reshape(-1,16*6)#axis->matrix->6d
        gt_shape_r = targets["mano.beta.r"]
        gt_cam_r = targets["mano.cam_t.wp.r"]
        gt_r = torch.cat([gt_pose_r, gt_shape_r, gt_cam_r], dim=-1)

        gt_pose_l = matrix_to_rotation_6d(targets["mano.pose.l"].reshape(-1,3,3)).reshape(-1,16*6)
        gt_shape_l = targets["mano.beta.l"]
        gt_cam_l = targets["mano.cam_t.wp.l"]
        gt_l = torch.cat([gt_pose_l, gt_shape_l, gt_cam_l], dim=-1)

        gt_rot_o = targets["object.rot"]
        gt_radian_o = targets["object.radian"]
        gt_cam_o = targets["object.cam_t.wp"]
        gt_o = torch.cat([gt_rot_o, gt_radian_o, gt_cam_o], dim=-1)
        
        x_0 = torch.cat([gt_r, gt_l, gt_o], dim=-1)
        
        return x_0

    def q_sample(self, x_0, t, noise):
        
        x_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 \
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        return x_t
    
    def init_eval(self):
        
        use_timesteps = set(space_timesteps(self.num_timesteps, self.timestep_respacing))
        self.timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        self.test_betas = np.array(new_betas)

        self.num_timesteps_test = int(self.test_betas.shape[0])

        test_alphas = 1.0 - self.test_betas
        self.test_alphas_cumprod = np.cumprod(test_alphas, axis=0)
        self.test_alphas_cumprod_prev = np.append(1.0, self.test_alphas_cumprod[:-1])
        self.test_alphas_cumprod_next = np.append(self.test_alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.testsqrt_alphas_cumprod = np.sqrt(self.test_alphas_cumprod)
        self.test_sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.test_alphas_cumprod)
        self.test_log_one_minus_alphas_cumprod = np.log(1.0 - self.test_alphas_cumprod)
        self.test_sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod)
        self.test_sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.test_alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.test_posterior_variance = (
                self.test_betas * (1.0 - self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.test_posterior_log_variance_clipped = np.log(
            np.append(self.test_posterior_variance[1], self.test_posterior_variance[1:])
        )
        self.test_posterior_mean_coef1 = (
                self.test_betas * np.sqrt(self.test_alphas_cumprod_prev) / (1.0 - self.test_alphas_cumprod)
        )
        self.test_posterior_mean_coef2 = (
                (1.0 - self.test_alphas_cumprod_prev)
                * np.sqrt(test_alphas)
                / (1.0 - self.test_alphas_cumprod)
        )
        
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                extract_into_tensor(self.test_posterior_mean_coef1, t, x_t.shape) * x_start
                + extract_into_tensor(self.test_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.test_posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.test_posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                extract_into_tensor(self.test_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / extract_into_tensor(self.test_sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def guidance_diffusion(self, pred, input, field):

        output = generic.prepare_kp_interfield(pred, max_dist=float("inf"))
        loss = diffusion_dist_loss_kp(output, field)
        model_output = self.output_process(pred)

        loss.backward()
        update = model_output - input.grad * self.scale

        return update
    
    def ddim_sample(self, x, ts, feature, meta_info, updated_field):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        time_emb = self.embed_timestep(new_ts)
        output = self.arctic_model(feature, meta_info, x, time_emb, updated_field)

        return output

    def ddim_sample_loop(self, x_t, feature, meta_data, updated_field, eta=0.0):
        indices = list(range(self.num_timesteps_test))[::-1]
        preds = []

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0], device=x_t.device)

            with torch.enable_grad():
                x_t = torch.autograd.Variable(x_t, requires_grad=True)
                pred = self.ddim_sample(x_t, t, feature, meta_data, updated_field)
                preds.append(pred.detach().to(x_t.device))
            
                #guidance diffusion
                pred_xstart = self.guidance_diffusion(pred, x_t, updated_field)

            # construct x_{t-1}
            # model_output = self.output_process(pred)

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x_t, t, pred_xstart)

            alpha_bar = extract_into_tensor(self.test_alphas_cumprod, t, x_t.shape)
            alpha_bar_prev = extract_into_tensor(self.test_alphas_cumprod_prev, t, x_t.shape)
            sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise = self.generate_noise(feature)
            mean_pred = (
                    pred_xstart * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            )  # no noise when t == 0
            x_t = mean_pred + nonzero_mask * sigma * noise

        return preds[-1]
    
    def forward(self, inputs, targets, meta_info):
        
        #feature extraction
        images = inputs["img"]#64,3,224,224
        features = self.backbone(images)[:,1:]#64,2048,7,7 for resnet, 64,197,768 for ViT
        features = features.permute(0,2,1).reshape(-1,self.feat_dim,self.num_patch,self.num_patch).contiguous()#64,196,768->64,768,196->64,768,14,14
    
        #field prediction
        updated_field = self.field_model(features, meta_info)

        #generate noise
        noise = self.generate_noise(features)
        
        if self.mode == "train":
            x_0 = self.input_process(targets)
            
            t, _ = self.sampler.sample(x_0.shape[0], x_0.device)
            
            time_emb = self.embed_timestep(t)

            x_t = self.q_sample(x_0, t, noise)
            
            #hand/object pose prediction
            output = self.arctic_model(features, meta_info, x_t, time_emb, updated_field)
            
        else:
            self.init_eval()
            
            output = self.ddim_sample_loop(noise, features, meta_info, updated_field)

        #recalculate field
        output = generic.prepare_kp_interfield(output, max_dist=float("inf"), alterkey=True)
        
        #merge output predicted field key
        output.update(updated_field)

        return output
