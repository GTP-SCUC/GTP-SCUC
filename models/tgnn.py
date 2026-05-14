import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class HeteroNodeProjection(nn.Module):
    """ Heterogeneous Node Projection (Eq. 5) """
    def __init__(self, raw_dim, hidden_dim):
        super().__init__()
        self.W_tau = nn.Linear(raw_dim, hidden_dim)
    
    def forward(self, X_raw_tau):
        # \tilde{X}_{t,\tau} = X_{t,\tau}^{raw} W_{\tau} + b_{\tau}
        return self.W_tau(X_raw_tau)

class TGNN(nn.Module):
    """
    Temporal Graph Neural Network for Spatiotemporal Feature Extraction.
    Refer to Section 3.3, Eq. 5-8.
    """
    def __init__(self, node_types, raw_dims, hidden_dim=128, gru_dim=64, num_heads=4):
        super(TGNN, self).__init__()
        
        # 1. Node Projection (Eq. 5)
        self.projections = nn.ModuleDict({
            ntype: HeteroNodeProjection(raw_dims[ntype], hidden_dim) for ntype in node_types
        })
        
        # 2. Spatial GAT (Eq. 6) - L=3 layers as per Section 4.3.2
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        
        # 3. Temporal GRU (Eq. 7)
        self.gru = nn.GRUCell(hidden_dim, gru_dim)
        
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim

    def forward(self, X_dict, edge_index, M_hat, h_prev):
        """
        X_dict: Dictionary of node features per type at time t
        M_hat: Dynamic Fault Mask \hat{M}_{i,t} \in {1, \kappa}
        h_prev: Hidden state from t-1
        """
        # Eq. 5: Heterogeneous alignment
        X_projected = []
        for ntype, X_raw in X_dict.items():
            X_projected.append(self.projections[ntype](X_raw))
        X_t = torch.cat(X_projected, dim=0) # Unified latent subspace
        
        # Eq. 6: Spatial Aggregation (GAT)
        h_spatial = self.gat1(X_t, edge_index)
        h_spatial = torch.relu(h_spatial)
        h_spatial = self.gat2(h_spatial, edge_index)
        h_spatial = torch.relu(h_spatial)
        h_spatial = self.gat3(h_spatial, edge_index)
        
        # Eq. 7: Temporal Evolution (GRU)
        h_t = self.gru(h_spatial, h_prev)
        
        # Eq. 8: Dynamic Fault Masking & Global Temporal Pooling
        # Apply high-sensitivity attention mask (M_hat)
        h_masked = h_t * M_hat 
        
        # E_t = Meanpooling({h_{i,t} \odot \hat{M}_{i,t}})
        E_t = torch.mean(h_masked, dim=0, keepdim=True) 
        
        return E_t, h_t
