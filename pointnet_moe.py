import torch
import torch.nn as nn
import pointnet_utils


class get_model(nn.Module):
    def __init__(self, weight_min, weight_max, phi_1, phi_2, buffer=5.0, normal_channel=True):
        """
        Args:
            weight_min: Minimum weight in the dataset.
            weight_max: Maximum weight in the dataset.
            phi_1: 33.3rd percentile of the weight distribution.
            phi_2: 66.7th percentile of the weight distribution.
            buffer: Boundary buffer to prevent sigmoid outputs from dying at 0.0 or 1.0.
            normal_channel: Whether the input incorporates normal channels.
        """
        super(get_model, self).__init__()

        # Feature extractor with 7 channels configured
        self.feat = pointnet_utils.PointNetEncoder(global_feat=True, feature_transform=False, channel=7)

        # Expert regression heads
        def create_expert():
            return nn.Sequential(
                nn.Linear(1027, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )

        self.expert_light = create_expert()
        self.expert_mid = create_expert()
        self.expert_heavy = create_expert()

        # Dynamic boundaries based on dataset statistics (min, max, 33%, 66%)
        # Buffer is added to keep the predictions within the effective gradient range of the Sigmoid function
        self.bound_light = [weight_min - buffer, phi_1 + buffer]
        self.bound_mid = [phi_1 - buffer, phi_2 + buffer]
        self.bound_heavy = [phi_2 - buffer, weight_max + buffer]

    def forward(self, x, bbox_attr):
        # Extract global spatial features
        feat_global, trans, trans_feat = self.feat(x)

        # Concatenate geometric descriptors
        combined_feat = torch.cat([feat_global, bbox_attr], dim=1)

        # Raw expert predictions
        raw_light = self.expert_light(combined_feat)
        raw_mid = self.expert_mid(combined_feat)
        raw_heavy = self.expert_heavy(combined_feat)

        # Sigmoid scaling bounded by dynamic physical intervals
        out_light = torch.sigmoid(raw_light) * (self.bound_light[1] - self.bound_light[0]) + self.bound_light[0]
        out_mid = torch.sigmoid(raw_mid) * (self.bound_mid[1] - self.bound_mid[0]) + self.bound_mid[0]
        out_heavy = torch.sigmoid(raw_heavy) * (self.bound_heavy[1] - self.bound_heavy[0]) + self.bound_heavy[0]

        return out_light, out_mid, out_heavy, trans_feat


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        # Use unreduced MSE Loss to apply expert-specific masking later
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, target_weight, target_cls, trans_feat):
        p0, p1, p2 = preds

        # Flatten tensors
        p0, p1, p2 = p0.view(-1), p1.view(-1), p2.view(-1)
        target_weight = target_weight.view(-1)

        # Calculate MSE loss for all experts across all samples
        loss0 = self.mse(p0, target_weight)
        loss1 = self.mse(p1, target_weight)
        loss2 = self.mse(p2, target_weight)

        # Generate directional penalty masks based on classification labels
        mask0 = (target_cls == 0).float()
        mask1 = (target_cls == 1).float()
        mask2 = (target_cls == 2).float()

        # Enforce local specialization: each expert is solely responsible for its designated domain
        final_loss = (loss0 * mask0 + loss1 * mask1 + loss2 * mask2).mean()

        return final_loss