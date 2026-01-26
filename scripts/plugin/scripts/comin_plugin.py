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


jg = 1  # set the domain id

## primary constructor
# request to register the variable
RHI_MAX_descriptor = ("RHI_MAX", jg)
QI_MAX_descriptor = ("QI_MAX", jg)
log_descriptor = ("log", jg)

comin.var_request_add(RHI_MAX_descriptor, lmodexclusive=False)
comin.var_request_add(QI_MAX_descriptor, lmodexclusive=False)
comin.var_request_add(log_descriptor, lmodexclusive=False)


domain = comin.descrdata_get_domain(jg)
domain_np = np.asarray(domain.cells.decomp_domain)


# Set metadata
comin.metadata_set(
    RHI_MAX_descriptor,
    zaxis_id=comin.COMIN_ZAXIS_2D,
    long_name="Maximum relative humidity over ice",
    units="%",
)

comin.metadata_set(
    QI_MAX_descriptor,
    zaxis_id=comin.COMIN_ZAXIS_2D,
    long_name="Maximum cloud ice content",
    units="kg/kg",
)

comin.metadata_set(
    log_descriptor,
    zaxis_id=comin.COMIN_ZAXIS_2D,
    long_name="Log file",
    units="",
)


## secondary constructor
@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def simple_python_constructor():
    global RHI_MAX, QI_MAX, temp, qv, exner, qi, log
    RHI_MAX = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        RHI_MAX_descriptor,
        flag=comin.COMIN_FLAG_WRITE,
    )
    QI_MAX = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        QI_MAX_descriptor,
        flag=comin.COMIN_FLAG_WRITE,
    )

    log = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        log_descriptor,
        flag=comin.COMIN_FLAG_WRITE,
    )

    temp = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", jg), flag=comin.COMIN_FLAG_READ
    )
    qv = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("qv", jg), flag=comin.COMIN_FLAG_READ
    )
    exner = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("exner", jg), flag=comin.COMIN_FLAG_READ
    )
    qi = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("qi", jg), flag=comin.COMIN_FLAG_READ
    )


## help function
# help function to calculate RHI_MAX and QI_MAX
def rhi(temp, qv, p_ex):
    import numpy as np

    rdv = (rd := 287.04) / (rv := 461.51)
    pres = (p0ref := 100000) * np.exp(((cpd := 1004.64) / rd) * np.ma.log(p_ex))
    e_s = 610.78 * np.ma.exp(21.875 * (temp - 273.15) / (temp - 7.66))
    e = pres * qv / (rdv + (1.0 - (rd / rv)) * qv)
    return 100.0 * e / e_s


## callback function
@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_AFTER)
def calculate_rhi_qi():
    # print("simple_python_callbackfct called!", file=sys.stderr)

    # create mask
    mask2d = domain_np != 0
    mask3d = np.repeat(mask2d[:, None, :], domain.nlev, axis=1)

    # apply mask to temp, qv, exner, qi
    temp_np = ma.masked_array(np.squeeze(temp), mask=mask3d)
    qv_np = ma.masked_array(np.squeeze(qv), mask=mask3d)
    exner_np = ma.masked_array(np.squeeze(exner), mask=mask3d)
    qi_np = ma.masked_array(np.squeeze(qi), mask=mask3d)

    # calculate RHI_MAX
    RHI_MAX_np = np.squeeze(RHI_MAX)
    RHI_MAX_3d = rhi(temp_np, qv_np, exner_np)
    RHI_MAX_np[:, :] = np.max(RHI_MAX_3d, axis=1)
    # print("RHI_MAX_np shape", RHI_MAX_np.shape, file=sys.stderr)

    # calculate QI_MAX
    QI_MAX_np = np.squeeze(QI_MAX)
    QI_MAX_np[:, :] = np.max(qi_np, axis=1)
    # print("QI_MAX_np shape", QI_MAX_np.shape, file=sys.stderr)


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
    
    RHI_MAX_np_glb = util_gather(np.asarray(RHI_MAX))
    QI_MAX_np_glb = util_gather(np.asarray(QI_MAX))

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
        total_size = RHI_MAX_np_glb.shape[0]

        # number of batches
        num_batches = total_size // one_batch_size

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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from checkpoint at {current_time_np}", file=sys.stderr)

        lossfunc = torch.nn.MSELoss()
        losses = []

        for i in range(num_batches):
            x_batch_np = RHI_MAX_np_glb[i * one_batch_size : (i + 1) * one_batch_size]
            y_batch_np = QI_MAX_np_glb[i * one_batch_size : (i + 1) * one_batch_size]

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

            loss.backward()

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

