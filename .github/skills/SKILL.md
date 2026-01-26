---
name: "MEssE Agent Skills Guide"
description: "A guide and reference to using AI agents effectively for the MEssE project."
---

# Skill Instructions
This skill helps you create scripts to build, run and couple the python scripts to ICON for the MEssE project, which amins to train a deep learning model to learn the essence of ICON simulations while these simulations are running. 


## When to use this skill

Use this skill when you need to :
- Search and understand the functions in ComIn plugin and related scripts
- Ceate specific environments for running ICON with the ComIn plugin on Levante
- Decide which tools, from ComIn, or YAC, or other libraries, to use for coupling ICON with the ML model.
- Debug issues related to building ICON with the ComIn plugin

## What are they:
### ICON
GitLab repository: https://gitlab.dkrz.de/icon/icon-model

ICON is a modeling framework for weather, climate, and environmental prediction.

### ComIn
GitHub repository: https://gitlab.dkrz.de/icon-comin/comin

ComIn organizes the data exchange and simulation events between the ICON model and “3rd party modules”.

### YAC
GitHub repository: https://dkrz-sw.gitlab-pages.dkrz.de/yac/

AYC A Coupling Library for Earth System Models


## Common Workflows

### Build

1. Build ICON `./scripts/build_scripts/build_icon.sh`

If the icon-model is not cloned, the scripts will automatically clone the icon-model from the github repository.

2. Build ComIn `./scripts/build_scripts/build_ComIn.sh`

3. Build YAC `./scripts/build_scripts/build_YAC.sh`

4. Build python environment `./scripts/build_scripts/python_env.sh`


### Run
1. make sure that no bug in plugin `./scripts/plugin/`

2. Prepare the input data `./scripts/run_icon/prepare_data.sh`

3. Run ICON `./scripts/run_icon/run_icon_LAM.sh`

## Files
`documents/ComIn`: Documentation related to ComIn plugin, with basic introduction.

`documents/YAC`: Documentation related to YAC library, with basic introduction and pythonic way to use the yac.

`examples/ComIn`: Examples of using ComIn plugin with ICON.

- `examples/ComIn/P1_exercise.ipynb`: example notebook for a rather simple comIn python Plogin. from which you can learn about mask array and combine data from MPI processes.
- `examples/ComIn/P2_exercise.ipynb`: from which you can learn the data structure of ICON in the memory, and how to defeine a mask for them using python.
- `examples/ComIn/P3_exercise.ipynb`: from which you can learn how run ICON with the python ComIn plugin.

`examples/YAC`: Examples of using YAC library to couple python code with ICON.