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
    global temp, tas, sfcwind
    temp = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", jg), flag=comin.COMIN_FLAG_READ
    )
    tas = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tas", jg), flag=comin.COMIN_FLAG_READ
    )
    sfcwind = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("sfcwind", jg), flag=comin.COMIN_FLAG_READ
    )

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


class Net(nn.Module):
    def __init__(self, n_inputs=30, n_outputs=30, n_hidden=32):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_inputs, n_hidden)  # 5*5 from image dimension
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_batch_callback():
    global net, optimizer, losses  # Add these as globals to persist across calls
    
    temp_np_glb = util_gather(np.asarray(temp))
    sfcwind_np_glb = util_gather(np.asarray(sfcwind))
    tas_np_glb = util_gather(np.asarray(tas))

    start_time = comin.descrdata_get_simulation_interval().run_start
    start_time_np = pd.to_datetime(start_time)

    current_time = comin.current_get_datetime()
    current_time_np = pd.to_datetime(current_time)

    if rank == 0:
        B = 5  # batch size
        C = 1  # channel
        H = 30  # height

        one_batch_size = B * C * H

        # total number of batches
        total_size = temp_np_glb.shape[0]

        # number of batches
        num_batches = total_size // one_batch_size

        # Validate input data for NaN/Inf
        if np.any(~np.isfinite(sfcwind_np_glb)) or np.any(~np.isfinite(tas_np_glb)):
            print("ERROR: Input data contains NaN or Inf values!", file=sys.stderr)
            print(f"  sfcwind_np_glb finite: {np.isfinite(sfcwind_np_glb).all()}", file=sys.stderr)
            print(f"  tas_np_glb finite: {np.isfinite(tas_np_glb).all()}", file=sys.stderr)
            return
        
        # ============================================
        # Data Normalization (Critical for stability!)
        # ============================================
        # Compute statistics for normalization
        x_mean, x_std = sfcwind_np_glb.mean(), sfcwind_np_glb.std()
        y_mean, y_std = tas_np_glb.mean(), tas_np_glb.std()
        
        # Avoid division by zero
        x_std = max(x_std, 1e-6)
        y_std = max(y_std, 1e-6)
        
        # Normalize data
        sfcwind_normalized = (sfcwind_np_glb - x_mean) / x_std
        tas_normalized = (tas_np_glb - y_mean) / y_std
        
        print(f"Data statistics:", file=sys.stderr)
        print(f"  sfcwind: mean={x_mean:.6f}, std={x_std:.6f}, range=[{sfcwind_np_glb.min():.6f}, {sfcwind_np_glb.max():.6f}]", file=sys.stderr)
        print(f"  tas:     mean={y_mean:.6f}, std={y_std:.6f}, range=[{tas_np_glb.min():.6f}, {tas_np_glb.max():.6f}]", file=sys.stderr)
        
        # Initialize model only at the first timestep
        if current_time_np == start_time_np:
            net = Net()
            learning_rate = 0.01
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
            print(f"Model initialized with random weights at {current_time_np}", file=sys.stderr)
        else:
            # Load model and optimizer state at subsequent timesteps
            checkpoint = torch.load(f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_np}.pth")
            if 'net' not in globals():  # Safety check in case model wasn't initialized
                net = Net()
                optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
            net.load_state_dict(checkpoint['model_state_dict'])
            # DO NOT load optimizer state - reset to avoid gradient explosion
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from checkpoint at {current_time_np}", file=sys.stderr)
            print(f"  Optimizer reset to avoid gradient explosion", file=sys.stderr)

        lossfunc = torch.nn.MSELoss()
        losses = []
        
        # Set model to training mode
        net.train()

        for i in range(num_batches):
            x_batch_np = sfcwind_normalized[i * one_batch_size : (i + 1) * one_batch_size]
            y_batch_np = tas_normalized[i * one_batch_size : (i + 1) * one_batch_size]

            # reshape
            x_batch_np = x_batch_np.reshape(B, C, H)
            y_batch_np = y_batch_np.reshape(B, C, H)

            # to tensor
            x_batch = torch.FloatTensor(x_batch_np)
            y_batch = torch.FloatTensor(y_batch_np)

            # train batch size (also) epochs
            optimizer.zero_grad()

            y_hat = net(x_batch)

            loss = lossfunc(y_hat, y_batch)
            
            # Check for NaN/Inf before backward pass
            if not torch.isfinite(loss):
                print(f"  WARNING: Loss is {loss.item()}, skipping batch {i}", file=sys.stderr)
                continue

            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            losses.append(loss.item())

            # print loss at this point
            print(f"loss: {loss.item()}", file=sys.stderr)

        # save state_dict
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_np}.pth")

        # print(f"model's state_dict saved at {current_time_np}", file=sys.stderr)

        # save log file
        with open(
            f"/scratch/{user[0]}/{user}/icon_exercise_comin/log_{current_time_np}.txt",
            "w",
        ) as f:
            for item in losses:
                f.write("%s\n" % item)
