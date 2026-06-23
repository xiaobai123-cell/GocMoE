import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, k=3, channel=7, normal_channel=False):
        super(get_model, self).__init__()

        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel)

        self.bbox_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(1024 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)

        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, bbox_attr):
        x, trans, trans_feat = self.feat(x)

        bbox_feat = self.bbox_mlp(bbox_attr)

        x = torch.cat([x, bbox_feat], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans_feat