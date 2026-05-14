import torch
import torch.nn as nn

class PhysicsConstrainedCGAN(nn.Module):
    """
    Conditional GAN with Physics-Constrained Loss for Extreme Scenario Augmentation.
    Refer to Section 3.2, Eq. 1-4 in the manuscript.
    """
    def __init__(self, condition_dim, noise_dim, output_dim):
        super(PhysicsConstrainedCGAN, self).__init__()
        
        # Generator: 3-layer MLP (1024, 512, 256) -> Linear De-projection
        # See Section 4.3.1 GAN parameter settings
        self.generator = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim)
        )
        
        # Discriminator: Symmetric downsampling (512, 256, 128)
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim + condition_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def generate(self, z, c):
        """ Generate scenario snapshots based on noise z and condition c. """
        x_input = torch.cat([z, c], dim=-1)
        return self.generator(x_input)

    def discriminate(self, x, c):
        """ Output authenticity probability. """
        d_input = torch.cat([x, c], dim=-1)
        return self.discriminator(d_input)

def physics_constraint_loss(P_hat, P_load, P_loss, P_avail_w, P_curt_w, P_avail_s, P_curt_s):
    """
    Calculates the physics constraint penalty (Eq. 4), mainly focusing on Power Balance (Eq. 3).
    """
    # Eq. 3: Power Balance MSE in the time domain
    total_gen = torch.sum(P_hat, dim=1) # Sum over generators N_g
    total_wind = torch.sum(P_avail_w - P_curt_w, dim=1)
    total_solar = torch.sum(P_avail_s - P_curt_s, dim=1)
    
    # Power balance violation
    power_imbalance = total_gen + total_wind + total_solar - P_load - P_loss
    C_power = torch.mean(power_imbalance ** 2)
    
    # Other constraints (Unit limits, Grid thermal limits, Ramping) - Implementations follow similarly
    C_unit, C_grid, C_temporal = 0.0, 0.0, 0.0 # Placeholder for operational limits
    
    # Eq. 4: Weighted composite loss (alpha=1.2, lambda1=0.3, lambda2=0.25, lambda3=0.3, lambda4=0.15)
    lambda_1, lambda_2, lambda_3, lambda_4 = 0.3, 0.25, 0.3, 0.15
    alpha = 1.2
    
    L_constraint = lambda_1 * C_power + lambda_2 * C_unit + lambda_3 * C_grid + lambda_4 * C_temporal
    return alpha * L_constraint
