# Model Essence Extractor (MEssE)

## Contributing

This repository uses branch protection on the main branch. All changes must go through pull requests and be reviewed by @liuquan18 before merging. See [.github/BRANCH_PROTECTION.md](.github/BRANCH_PROTECTION.md) for details.

# Structure
```bash
.
├── build                       # external location for building ICON
├── data                        # output of icon
├── environment                 # Environment setup scripts
├── experiment                  # Experiment data 
├── ICON                        # ICON source code
└── scripts                      # Scripts 
    ├── build_icon              # scripts for building ICON, ComIn, YAC 
    ├── plugin                  # Python scripts for ComIn     
    └── run_icon                # scripts for running ICON
```
# Workflow

## Build

1. Build ICON `./scripts/build_scripts/build_icon.sh`

If the icon-model is not cloned, the scripts will automatically clone the icon-model from the github repository.

2. Build ComIn `./scripts/build_scripts/build_ComIn.sh`

3. Build YAC `./scripts/build_scripts/build_YAC.sh`

4. Build python environment `./scripts/build_scripts/python_env.sh`


## Run
1. make sure that no bug in plugin `./scripts/plugin/`

2. Prepare the input data `./scripts/run_icon/prepare_data.sh`

3. Run ICON `./scripts/run_icon/run_icon_LAM.sh`
