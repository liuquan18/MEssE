""" 
This is a ComIn Python plugin designed for use in the ICON 2024 training course.
"""

# %%
import comin
import sys
#%%
import numpy as np
import numpy.ma as ma
import pandas as pd

# from datetime import datetime


# %%
import torch

import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI

import getpass
user = getpass.getuser()
torch.manual_seed(0)

# test start
glob = comin.descrdata_get_global()
n_dom=glob.n_dom
# make n_dom as np array
n_dom = np.array(n_dom)
print("number of domains:", n_dom, file=sys.stderr)
# test end


jg = 1  # set the domain id


domain = comin.descrdata_get_domain(jg)
domain_np = np.asarray(domain.cells.decomp_domain)



## secondary constructor
@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def simple_python_constructor():
    global pr
    # Try common precipitation variable names
    try:
        pr = comin.var_get(
            [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("pr", jg), flag=comin.COMIN_FLAG_READ
        )
        print("Using precipitation variable: pr", file=sys.stderr)
    except:
        try:
            pr = comin.var_get(
                [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("prec_rate", jg), flag=comin.COMIN_FLAG_READ
            )
            print("Using precipitation variable: prec_rate", file=sys.stderr)
        except:
            try:
                pr = comin.var_get(
                    [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tot_prec", jg), flag=comin.COMIN_FLAG_READ
                )
                print("Using precipitation variable: tot_prec", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Could not find precipitation variable. Tried: pr, prec_rate, tot_prec", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)
                raise

# help function to collect the data from all processes
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()

def util_gather(data_array: np.ndarray, root=0):

    # 0-shifted global indices for all local cells (including halo cells):
    global_idx = np.asarray(domain.cells.glb_index) - 1

    # no. of local cells (incl. halos):
    nc = domain.cells.ncells

    # to remove empty cells
    data_array_1d = data_array.ravel("F")[0:nc]
    decomp_domain_np_1d = domain_np.ravel("F")[0:nc]
    halo_mask = decomp_domain_np_1d == 0

    # to remove halo cells
    data_array_1d = data_array_1d[halo_mask]
    global_idx = global_idx[halo_mask]

    # gather operation
    data_buf = comm.gather((data_array_1d, global_idx), root=root)

    # reorder received data according to global_idx
    if rank == root:
        nglobal = sum([len(gi) for _, gi in data_buf])
        global_array = np.zeros(nglobal, dtype=np.float64)
        for data_array_i, global_idx_i in data_buf:
            global_array[global_idx_i] = data_array_i
        return global_array
    else:
        return None


class ConditionalDiffusionDownscaler(nn.Module):
    def __init__(self, n_fine_cells=256, n_neighbors=7, n_hidden=256, n_timesteps=50):
        """
        Conditional diffusion model for precipitation downscaling.
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

def build_neighbor_map(n_coarse=80):
    """
    Build hexagonal neighbor relationships for ICON R2B01 grid.
    For now, uses a simple approximation. In production, this should use
    domain.cells.neighbor_idx from ICON grid metadata.
    
    Args:
        n_coarse: Number of coarse cells (80 for R2B01)
    Returns:
        neighbor_map: (n_coarse, 6) array of neighbor indices
    """
    # Temporary placeholder: cyclic neighbors
    # TODO: Replace with actual ICON grid topology
    neighbor_map = np.zeros((n_coarse, 6), dtype=int)
    for i in range(n_coarse):
        neighbor_map[i] = [(i + j) % n_coarse for j in [1, 2, 3, -1, -2, -3]]
    return neighbor_map


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_batch_callback():
    global net, optimizer, neighbor_map, training_counter
    
    # Gather fine-resolution precipitation (R2B04: 20480 cells)
    pr_np_glb = util_gather(np.asarray(pr))

    start_time = comin.descrdata_get_simulation_interval().run_start
    start_time_np = pd.to_datetime(start_time)
    # Convert to ISO format string without spaces for filenames
    start_time_str = start_time_np.strftime('%Y-%m-%dT%H-%M-%S')

    current_time = comin.current_get_datetime()
    current_time_np = pd.to_datetime(current_time)
    # Convert to ISO format string without spaces for filenames
    current_time_str = current_time_np.strftime('%Y-%m-%dT%H-%M-%S')

    if rank == 0:
        # Constants for ICON grid structure
        N_R2B04 = 20480  # Fine resolution
        N_R2B01 = 80     # Coarse resolution
        N_FINE_PER_COARSE = N_R2B04 // N_R2B01  # 256 fine cells per coarse cell
        N_NEIGHBORS = 7  # Center + 6 neighbors
        
        # Validate dimensions
        if pr_np_glb.shape[0] != N_R2B04:
            print(f"ERROR: Expected {N_R2B04} cells but got {pr_np_glb.shape[0]}", file=sys.stderr)
            return
        
        # Validate input data for NaN/Inf
        if np.any(~np.isfinite(pr_np_glb)):
            print("ERROR: Precipitation contains NaN or Inf values!", file=sys.stderr)
            return
        
        # ============================================
        # Coarsen R2B04 → R2B01 (average 256 cells → 1)
        # ============================================
        pr_r2b04_fine = pr_np_glb.reshape(N_R2B01, N_FINE_PER_COARSE)  # Shape: (80, 256)
        pr_r2b01_coarse = pr_r2b04_fine.mean(axis=1)  # Shape: (80,) - average each region
        
        # ============================================
        # Build Spatial Context (Center + Neighbors)
        # ============================================
        if 'neighbor_map' not in globals():
            neighbor_map = build_neighbor_map(N_R2B01)
            print(f"Built neighbor map for {N_R2B01} coarse cells", file=sys.stderr)
        
        # Create context array: (80, 7) = [center, neighbor1, ..., neighbor6]
        pr_coarse_context = np.zeros((N_R2B01, N_NEIGHBORS))
        pr_coarse_context[:, 0] = pr_r2b01_coarse  # Center cell
        for i in range(N_R2B01):
            pr_coarse_context[i, 1:] = pr_r2b01_coarse[neighbor_map[i]]
        
        # ============================================
        # Data Normalization with EMA
        # ============================================
        # Current batch statistics (use center values only for statistics)
        current_coarse_mean = pr_r2b01_coarse.mean()
        current_coarse_std = pr_r2b01_coarse.std()
        current_fine_mean = pr_np_glb.mean()
        current_fine_std = pr_np_glb.std()
        
        print(f"Conditional diffusion statistics at {current_time_np}:", file=sys.stderr)
        print(f"  Current - Coarse: mean={current_coarse_mean:.6e}, std={current_coarse_std:.6e}, range=[{pr_r2b01_coarse.min():.6e}, {pr_r2b01_coarse.max():.6e}]", file=sys.stderr)
        print(f"  Current - Fine:   mean={current_fine_mean:.6e}, std={current_fine_std:.6e}, range=[{pr_np_glb.min():.6e}, {pr_np_glb.max():.6e}]", file=sys.stderr)
        
        # ============================================
        # Model Initialization
        # ============================================
        BUFFER_SIZE = 48  # Keep last 48 simulation timesteps = 12 hours (48 × 15min = 12h)
        N_DIFFUSION_STEPS = 50  # Diffusion denoising steps (50 for good quality/speed balance)
        
        # Check if this is the first callback (checkpoint doesn't exist yet)
        import os
        checkpoint_path = f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_str}.pth"
        is_first_run = not os.path.exists(checkpoint_path)
        
        if is_first_run:
            # First callback: initialize model
            net = ConditionalDiffusionDownscaler(
                n_fine_cells=N_FINE_PER_COARSE, 
                n_neighbors=N_NEIGHBORS,
                n_hidden=256, 
                n_timesteps=N_DIFFUSION_STEPS
            )
            learning_rate = 0.0001
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            
            # Initialize EMA statistics with first batch
            ema_coarse_mean = current_coarse_mean
            ema_coarse_std = max(current_coarse_std, 1e-6)
            ema_fine_mean = current_fine_mean
            ema_fine_std = max(current_fine_std, 1e-6)
            
            # Initialize experience replay buffer (stores normalized data)
            replay_buffer = []
            
            # Initialize training counter (independent of simulation time)
            training_counter = 0
            
            print(f"Conditional diffusion model initialized at {current_time_np}", file=sys.stderr)
            print(f"  Architecture: {N_NEIGHBORS} coarse (context) → {N_FINE_PER_COARSE} fine (via diffusion)", file=sys.stderr)
            print(f"  Diffusion steps: {N_DIFFUSION_STEPS}, Hidden dim: 256", file=sys.stderr)
            print(f"  Learning rate: {learning_rate}, Buffer size: {BUFFER_SIZE} timesteps (12h)", file=sys.stderr)
            print(f"  Training: Starts when buffer is full, then trains every callback with rolling window, 3 epochs per training, EMA momentum: 0.95", file=sys.stderr)
            print(f"  EMA stats - Coarse: mean={ema_coarse_mean:.6e}, std={ema_coarse_std:.6e}", file=sys.stderr)
            print(f"  EMA stats - Fine:   mean={ema_fine_mean:.6e}, std={ema_fine_std:.6e}", file=sys.stderr)
        else:
            # Load model and statistics from previous checkpoint
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if 'net' not in globals():
                net = ConditionalDiffusionDownscaler(
                    n_fine_cells=N_FINE_PER_COARSE,
                    n_neighbors=N_NEIGHBORS,
                    n_hidden=256,
                    n_timesteps=N_DIFFUSION_STEPS
                )
                optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
            
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load and update EMA statistics
            ema_momentum = 0.95
            ema_coarse_mean = ema_momentum * checkpoint['ema_coarse_mean'] + (1 - ema_momentum) * current_coarse_mean
            ema_coarse_std = ema_momentum * checkpoint['ema_coarse_std'] + (1 - ema_momentum) * max(current_coarse_std, 1e-6)
            ema_fine_mean = ema_momentum * checkpoint['ema_fine_mean'] + (1 - ema_momentum) * current_fine_mean
            ema_fine_std = ema_momentum * checkpoint['ema_fine_std'] + (1 - ema_momentum) * max(current_fine_std, 1e-6)
            
            # Load replay buffer and training counter
            replay_buffer = checkpoint.get('replay_buffer', [])
            training_counter = checkpoint.get('training_counter', 0)
            
            print(f"Model loaded from checkpoint at {current_time_np}", file=sys.stderr)
            print(f"  EMA stats - Coarse: mean={ema_coarse_mean:.6e}, std={ema_coarse_std:.6e}", file=sys.stderr)
            print(f"  EMA stats - Fine:   mean={ema_fine_mean:.6e}, std={ema_fine_std:.6e}", file=sys.stderr)
            print(f"  Replay buffer: {len(replay_buffer)} timesteps in memory", file=sys.stderr)
            print(f"  Training sessions completed: {training_counter}", file=sys.stderr)
        
        # ============================================
        # Normalize current data with EMA statistics
        # ============================================
        pr_coarse_context_norm = (pr_coarse_context - ema_coarse_mean) / ema_coarse_std
        pr_fine_norm = (pr_r2b04_fine - ema_fine_mean) / ema_fine_std
        
        # ============================================
        # Experience Replay Buffer (Always Update)
        # ============================================
        # Add current timestep to buffer (context + fine field)
        current_samples = (pr_coarse_context_norm.copy(), pr_fine_norm.copy())
        replay_buffer.append(current_samples)
        
        # Keep only last BUFFER_SIZE timesteps
        if len(replay_buffer) > BUFFER_SIZE:
            replay_buffer.pop(0)
        
        # ============================================
        # Check if buffer is full before training
        # ============================================
        if len(replay_buffer) < BUFFER_SIZE:
            # Buffer not full yet, save checkpoint and skip training
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_coarse_mean': ema_coarse_mean,
                'ema_coarse_std': ema_coarse_std,
                'ema_fine_mean': ema_fine_mean,
                'ema_fine_std': ema_fine_std,
                'replay_buffer': replay_buffer,
                'training_counter': training_counter,
            }, f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_str}.pth")
            print(f"Buffer filling: {len(replay_buffer)}/{BUFFER_SIZE} timesteps collected (training starts when full)", file=sys.stderr)
            return
        
        # ============================================
        # Buffer is full - train with all buffered samples
        # ============================================
        
        # ============================================
        # Buffer is full - train with all buffered samples
        # ============================================
        # Combine all buffered samples for training
        all_context = np.concatenate([x for x, y in replay_buffer], axis=0)  # Shape: (N_buffered * 80, 7)
        all_fine = np.concatenate([y for x, y in replay_buffer], axis=0)     # Shape: (N_buffered * 80, 256)
        
        n_total_samples = all_context.shape[0]
        training_counter += 1
        print(f"  Training session {training_counter} at {current_time_np}: {len(replay_buffer)} timesteps in buffer = {n_total_samples} samples", file=sys.stderr)
        
        lossfunc = torch.nn.MSELoss()
        net.train()

        # ============================================
        # Training: Diffusion Model with Random Denoising Timesteps
        # ============================================
        N_EPOCHS = 1  # Single epoch for continuous training (model trains every 15 minutes)
        BATCH_SIZE = 16
        
        # Prepare data tensors from accumulated buffer
        x_context = torch.FloatTensor(all_context)  # Shape: (N_total, 7)
        x_fine = torch.FloatTensor(all_fine)        # Shape: (N_total, 256)
        
        epoch_losses = []
        
        for epoch in range(N_EPOCHS):
            # Shuffle all samples for each epoch
            indices = torch.randperm(n_total_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_total_samples, BATCH_SIZE):
                batch_idx = indices[i:i+BATCH_SIZE]
                context_batch = x_context[batch_idx]  # Shape: (batch_size, 7)
                fine_batch = x_fine[batch_idx]        # Shape: (batch_size, 256)
                
                # Random diffusion timestep for each sample
                t = torch.randint(0, N_DIFFUSION_STEPS, (context_batch.shape[0],), dtype=torch.long)
                
                # Forward diffusion: add noise to fine field
                x_noisy, true_noise = net.forward_diffusion(fine_batch, t)
                
                optimizer.zero_grad()
                
                # Predict noise
                predicted_noise = net(x_noisy, context_batch, t)
                
                # Loss: difference between predicted and true noise
                loss = lossfunc(predicted_noise, true_noise)
                
                if not torch.isfinite(loss):
                    print(f"  WARNING: NaN/Inf loss at epoch {epoch}, batch {i//BATCH_SIZE}, skipping", file=sys.stderr)
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Calculate average loss for this epoch
            if n_batches > 0:
                avg_epoch_loss = epoch_loss / n_batches
                epoch_losses.append(avg_epoch_loss)
                print(f"    Epoch {epoch+1}/{N_EPOCHS}: avg loss = {avg_epoch_loss:.6f} ({n_batches} batches)", file=sys.stderr)
            else:
                print(f"    Epoch {epoch+1}/{N_EPOCHS}: no valid batches", file=sys.stderr)
        
        final_loss = epoch_losses[-1] if epoch_losses else float('nan')
        print(f"  Training session {training_counter} complete: final loss = {final_loss:.6f}", file=sys.stderr)
        
        # ============================================
        # Save checkpoint and logs
        # ============================================
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_coarse_mean': ema_coarse_mean,
            'ema_coarse_std': ema_coarse_std,
            'ema_fine_mean': ema_fine_mean,
            'ema_fine_std': ema_fine_std,
            'replay_buffer': replay_buffer,
            'training_counter': training_counter,
        }, f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_str}.pth")

        # Save all epoch losses with training counter for consistent tracking
        with open(
            f"/scratch/{user[0]}/{user}/icon_exercise_comin/log_session_{training_counter:04d}_{current_time_str}.txt",
            "w",
        ) as f:
            f.write(f"training_session: {training_counter}\n")
            f.write(f"simulation_time: {current_time_str}\n")
            for i, loss_val in enumerate(epoch_losses):
                f.write(f"epoch_{i+1}: {loss_val:.6f}\n")