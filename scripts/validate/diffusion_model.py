"""
Standalone diffusion model definition for testing.
This is a copy of the ConditionalDiffusionDownscaler class without comin dependencies.
"""

import torch
import torch.nn as nn


class ConditionalDiffusionDownscaler(nn.Module):
    def __init__(self, n_fine_cells=256, n_neighbors=7, n_hidden=256, n_timesteps=50):
        """
        Conditional diffusion model for near-surface air temperature downscaling.
        Generates diverse fine-resolution fields conditioned on coarse spatial context.
        
        Args:
            n_fine_cells: Number of fine cells per coarse cell (256 for R2B04/R2B01)
            n_neighbors: Spatial context size (1 center + 6 neighbors = 7)
            n_hidden: Hidden layer dimension
            n_timesteps: Number of diffusion steps (50 for good quality)
        """
        super(ConditionalDiffusionDownscaler, self).__init__()
        
        self.n_timesteps = n_timesteps
        self.n_fine_cells = n_fine_cells
        
        # ===== Noise Schedule (Linear) =====
        betas = torch.linspace(0.0001, 0.02, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # ===== Context Encoder (Spatial Information) =====
        self.context_encoder = nn.Sequential(
            nn.Linear(n_neighbors, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        
        # ===== Timestep Embedding =====
        self.time_mlp = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden)
        )
        
        # ===== Denoising Network =====
        # Input: noisy_field + context_embedding + time_embedding
        self.denoise_net = nn.Sequential(
            nn.Linear(n_fine_cells + n_hidden * 2, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_fine_cells)  # Predict noise
        )
    
    def get_timestep_embedding(self, t, dim=128):
        """
        Sinusoidal timestep embedding (like Transformer positional encoding).
        
        Args:
            t: (batch_size,) timestep indices
            dim: embedding dimension
        Returns:
            (batch_size, dim) embeddings
        """
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward_diffusion(self, x_0, t):
        """
        Add noise to clean data (training only).
        
        Args:
            x_0: (batch, n_fine_cells) - clean fine-resolution field
            t: (batch,) - timestep indices [0, n_timesteps)
        Returns:
            x_t: (batch, n_fine_cells) - noisy field
            noise: (batch, n_fine_cells) - added noise
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise
    
    def forward(self, x_t, context, t):
        """
        Predict noise given noisy field, spatial context, and timestep.
        
        Args:
            x_t: (batch, n_fine_cells) - noisy fine field
            context: (batch, n_neighbors) - coarse values [center + 6 neighbors]
            t: (batch,) - timestep indices
        Returns:
            predicted_noise: (batch, n_fine_cells)
        """
        # Encode spatial context
        context_emb = self.context_encoder(context)  # (batch, n_hidden)
        
        # Encode timestep
        t_emb = self.get_timestep_embedding(t, dim=context_emb.shape[1])
        t_emb = self.time_mlp(t_emb)  # (batch, n_hidden)
        
        # Concatenate all information
        x_input = torch.cat([x_t, context_emb, t_emb], dim=1)
        
        # Predict noise
        predicted_noise = self.denoise_net(x_input)
        return predicted_noise
    
    @torch.no_grad()
    def sample(self, context, n_samples=1):
        """
        Generate fine-resolution fields from coarse context (inference).
        
        Args:
            context: (batch, n_neighbors) - coarse values
            n_samples: number of samples to generate per context
        Returns:
            samples: (batch, n_samples, n_fine_cells)
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Start from pure Gaussian noise
        x_t = torch.randn(batch_size, n_samples, self.n_fine_cells, device=device)
        
        # Iteratively denoise from t=T-1 to t=0
        for t_idx in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            
            # Expand context and timestep for n_samples
            context_exp = context[:, None, :].expand(-1, n_samples, -1).reshape(batch_size * n_samples, -1)
            x_t_flat = x_t.reshape(batch_size * n_samples, -1)
            t_exp = t[:, None].expand(-1, n_samples).reshape(-1)
            
            # Predict noise
            predicted_noise = self.forward(x_t_flat, context_exp, t_exp)
            predicted_noise = predicted_noise.reshape(batch_size, n_samples, -1)
            
            # Denoise step
            alpha_t = self.alphas[t_idx]
            alpha_cumprod_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]
            
            # x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_t)
            x_t = (x_t - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Add noise (except at final step)
            if t_idx > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(beta_t) * noise
        
        return x_t
