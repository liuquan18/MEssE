#!/usr/bin/env python3

from yac import YAC
import yaml

yac = YAC()

yac.def_comps([])

yac.enddef()

metadata = {comp: {"metadata": yaml.safe_load(yac.get_component_metadata(comp) or ""),
                   "grids": {
                       grid: {"metadata": yaml.safe_load(yac.get_grid_metadata(grid) or ""),
                              "fields": {
                                  field: (yaml.safe_load(yac.get_field_metadata(comp, grid, field) or "") or {}) |
                                  {"collection_size": yac.get_field_collection_size(comp, grid, field),
                                   "timestep": yac.get_field_timestep(comp, grid, field),
                                   "role": yac.get_field_role(comp, grid, field).name}
                                  for field in yac.get_field_names(comp, grid)}
                              } for grid in yac.get_comp_grid_names(comp)}
                   } for comp in yac.component_names}

with open("metadata.yaml", "w") as f:
    yaml.dump(metadata, f)
