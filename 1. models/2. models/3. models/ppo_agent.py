import torch
import torch.nn as nn

class DASTActor(nn.Module):
    """
    PPO Actor with Dynamic Affine Scaling Transformation (DAST) for SCUC Hybrid Action Space.
    Refer to Section 3.4, Eq. 9-11.
    """
    def __init__(self, emb_dim, num_units):
        super(DASTActor, self).__init__()
        
        # 4-layer fully connected network (512, 256, 128, 64) with GELU
        self.shared_mlp = nn.Sequential(
            nn.Linear(emb_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU()
        )
        
        # Discrete Branch (Unit Commitment Status)
        self.status_head = nn.Sequential(
            nn.Linear(64, num_units),
            nn.Sigmoid() # Outputs \pi_{i,t}^{status}
        )
        
        # Continuous Branch (Power Dispatch Mean)
        self.power_head = nn.Sequential(
            nn.Linear(64, num_units),
            nn.Sigmoid() # Base normalized output
        )
        
        # Log standard deviation for exploration in continuous space
        self.log_std = nn.Parameter(torch.zeros(num_units))

    def forward(self, E_t, P_min_dynamic, P_max_dynamic):
        """
        E_t: Spatiotemporal embedding from TGNN
        P_min_dynamic, P_max_dynamic: Bounds derived from ramping (Eq. 9, Eq. 10)
        """
        features = self.shared_mlp(E_t)
        
        # 1. Discrete action probability (\pi_{i,t}^{status})
        pi_status = self.status_head(features)
        
        # 2. Continuous action with DAST (Eq. 11)
        raw_power = self.power_head(features)
        # Dynamic Affine Scaling Transformation: limits space to dynamic varying bounds
        mu_power = P_min_dynamic + raw_power * (P_max_dynamic - P_min_dynamic)
        
        return pi_status, mu_power, self.log_std

class PPOCritic(nn.Module):
    """
    PPO Critic network for value estimation V(s_t). (Eq. 12)
    """
    def __init__(self, emb_dim):
        super(PPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, E_t):
        return self.net(E_t)
