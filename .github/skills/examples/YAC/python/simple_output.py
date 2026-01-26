#!/usr/bin/env python3

from yac import YAC, Reg2dGrid, Location, Field, TimeUnit, InterpolationStack, NNNReductionType, Action
from netCDF4 import Dataset
import numpy as np
import argparse
import isodate
import cftime
import sys

parser = argparse.ArgumentParser(prog="simple_output.py")
parser.add_argument("filename", type=str, help="output filename")
parser.add_argument("src", type=str, nargs=3, help="(component, grid, field) tuple of the source field")
parser.add_argument("--res", type=int, nargs=2, help="resolution for lon-lat grid",
                    default=(360, 180))
args = parser.parse_args()

yac = YAC()

comp_name = f"simple_output_{args.src}"
comp = yac.def_comp(comp_name)

lon = np.linspace(-np.pi, np.pi, args.res[0], endpoint=False)
lat = np.linspace(-0.5*np.pi, 0.5*np.pi, args.res[1])
grid_name = f"simple_output_grid_{args.src}"
grid = Reg2dGrid(grid_name, lon, lat, cyclic=[True, False])
clon = lon+0.5*(lon[1]-lon[0])
clat = 0.5*(lat[1:]+lat[:-1])
points = grid.def_points(Location.CELL, clon, clat)

yac.sync_def()

collection_size = yac.get_field_collection_size(*args.src)
assert collection_size == 1
dt = yac.get_field_timestep(*args.src)
print(f"Timestep of {args.src}: {dt}", file=sys.stderr)
field = Field.create(args.src[2], comp, points, collection_size, dt, TimeUnit.ISO_FORMAT)

hcsbb = InterpolationStack()
hcsbb.add_hcsbb()
hcsbb.add_fixed(-1.0)

lag = 1
yac.def_couple(*args.src,
               comp_name, grid_name, args.src[2],
               dt, TimeUnit.ISO_FORMAT, 0, hcsbb,
               tgt_lag=lag)

yac.enddef()

dataset = Dataset(args.filename, "w")

start = isodate.parse_datetime(yac.start_datetime)
end = isodate.parse_datetime(yac.end_datetime)
timestep = isodate.parse_duration(dt)
no_timesteps = int((end-start)/timestep)+1-lag
print(f"{no_timesteps=}", file=sys.stderr)
time_dim = dataset.createDimension("time", no_timesteps)
time_var = dataset.createVariable("time", "i8", ("time",))
time_var.units = "seconds since "+yac.start_datetime
time_range = [start+i*timestep for i in range(no_timesteps)]
time_var[:] = cftime.date2num(time_range, units=time_var.units)

dataset.createDimension("clat", len(clat))
dataset.createDimension("clon", len(clon))
var = dataset.createVariable(args.src[2], "f4", ("time", "clat", "clon"))

data = None
for t in range(no_timesteps):
    print(f"Writing {field.name} at {field.datetime} ({t})", file=sys.stderr)
    data, info = field.get(data)
    var[t, :] = data.reshape((len(clat), len(clon)))
print("Finish!", file=sys.stderr)
