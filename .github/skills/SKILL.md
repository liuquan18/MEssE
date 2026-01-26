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

