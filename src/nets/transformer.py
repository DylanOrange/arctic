import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Transformer Fusion Module with Cross-Attention and Single Fused Feature
class Transformer(nn.Module):
    def __init__(self, image_feature_dim, point_cloud_feature_dim, num_heads, hidden_dim):
        super(Transformer, self).__init__()

        # Linear projections for image and point cloud features
        self.image_projection = nn.Linear(image_feature_dim, hidden_dim)
        self.point_cloud_projection = nn.Linear(point_cloud_feature_dim, hidden_dim)

        # Multi-Head Self-Attention for Image Features
        self.image_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first = True)

        # Multi-Head Self-Attention for Point Cloud Features
        self.point_cloud_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first = True)

        # Feed-Forward Networks
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, image_feature_dim),
            nn.ReLU(),
            nn.Linear(image_feature_dim, image_feature_dim)
        )
        
        # self.point_cloud_ffn = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image_features, point_cloud_features):
        #
        B, N, h, w = image_features.shape
        image_features = image_features.view(B,N,-1).permute(0,2,1)
        point_cloud_features = point_cloud_features.permute(0,2,1)

        # Project image and point cloud features to a common hidden dimension
        image_proj = self.image_projection(image_features)
        point_cloud_proj = self.point_cloud_projection(point_cloud_features)

        # Compute self-attended representations for image and point cloud features
        image_self_attention_output, _ = self.image_attention(image_proj, image_proj, image_proj)
        point_cloud_self_attention_output, _ = self.point_cloud_attention(point_cloud_proj, point_cloud_proj, point_cloud_proj)

        # Cross-Attention: Point Cloud Features attending to Image Features
        # point_cloud_to_image_attention_output, _ = self.image_attention(
        #     point_cloud_self_attention_output, image_proj, image_proj
        # )

        # Cross-Attention: Image Features attending to Point Cloud Features
        image_to_point_cloud_attention_output, _ = self.point_cloud_attention(
            image_self_attention_output, point_cloud_self_attention_output, point_cloud_self_attention_output
        )

        # Combine Cross-Attention Results
        combined_features = self.layer_norm(image_to_point_cloud_attention_output)

        # Feed-Forward Network
        fused_feature = self.image_ffn(combined_features)

        fused_feature = fused_feature.permute(0,2,1).view(B,N,h,w)

        return fused_feature