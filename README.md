# Project Styx 



## Table of content

- [Project description](#project-description)
- [Installation](#installation)
    * [Conda](#conda)
    * [pip](#pip)
- [Usage](#usage)
- [Contact](#contact)

## Project description

This repository contains code pertaining to the optimization of MIPsolvers
for neural network verification.

## Installation

### External tools
The following external tools need to be installed:

- [Gurobi](https://www.gurobi.com/) 
- [MIPVerify](https://github.com/vtjeng/MIPVerify.jl)

It is also highly recommended to compile the runsolver
from source on your own system. 
C++ is required to successfully compile.

- [runsolver](https://www.cril.univ-artois.fr/en/software/runsolver/)

### Conda
Tho create a conda environment with required packages 
install the ``environment.yml`` file using:

```
conda env create -f environment.yml
```

Then the enironment can be activated using:
```
conda activate Styx
```

### pip
To install all dependencies in a virtual environment,
use the ``requirements.txt`` file:

````
pip install -r requirements.txt
````


## Usage


Instances used for this project can be found at: `https://github.com/marti-mcfly/nn-verification-configuration.`.

### .env file
All algorithms except for SMAC2 use the .env file to change settings. 
The .env contains the following variables:

| Option                        | Purpose                                                       |   
|-------------------------------|---------------------------------------------------------------|
| INSTANCE_DIR                  | Directory with all instances                                  |
| PARAM_FILE                    | Absolute path to the .pcs file describing the parameter space |
| RUNSOLVER_LOC                 | The absolute path to the runsolver                            |
| ORACLE_CAPTIME                | Captime to use for Gurobi                                     |
| K0_UTILITY                    | k0 in selected utility function                               |
| K1_UTILITY                    | k1 in selected utility function                               |
| MAX_PROCESSES                 | Maximum number of parallel processes to run                   |
| BINARY_PATH                   | Path to Gurobi binary                                         |
| TRAIN_INSTANCE_DIR            | Directory containing training instances                       | 
| OBJECTIVE                     | "UTILITY" or "PAR"                                            |
| CASMO_RESPLACE_DOE_BY_DEFAULT | "TRUE" or "FALSE", replaced DoE for Casmopolitan              |
| SMAC_SEED                     | Random for the optimizer to use                               |

### SMAC2
To run the experiments of SMAC2, clone the aclib library at `https://bitbucket.org/mlindauer/aclib2/src/master/` into 
this repository. Proceed to use configuration files found in the same repository as the instances and run:
````
python3 aclib2/aclib/run.py -s nn-verification_mipverify -c SMAC2 -n 1 --env local --init_run_number {seed}
````

### SMAC3
Running SMAC3 can be done using the following command:
```
python3 SMAC3/smac3_runner.py ../aclib2/target_algorithms/sat/spear/MipVerify ./aclib2/instances/mip/data/SDPdMLPa-MIPVerify --runid={run_name} --u="u_geometric k0={k0} k1={k1}" --max_solver_time=9600 --overwrite
```

### COUP
```
python3 COUP_runner.py
```

### COUP+
```
python3 COUP+/coup_runner.py ../aclib2/target_algorithms/sat/spear/MipVerify ./aclib2/instances/mip/data/SDPdMLPa-MIPVerify --runid=coup_mipverify --n0=10 --k0=500 --u="u_geometric k0=250 k1=9600" --savemod 1
```

### Casmopolitan
Running Casmopolitan can be done using the following command:
```
python3 CASMOPOLITAN/main.py -p MIP --max_iter 100 --n_init 20 --n_trials 1
```

### Bounce
```
python3 Bounce/main.py --gin-files Bounce/configs/mipvery_alice.gin
```


## Plots
All plots can used can be found and generated using the .ipynb files in this repository.

The following outlines which notebooks contain the files for different plots and tables.

| Figure/Table | Notebook                                               |
|--------------|--------------------------------------------------------|
| Figure 1     | `compare_smac3.ipynb`                                  |
| Figure 2     | `compare.ipynb` & `compare_smac3.ipynb`                |
| Figure 3     | `compare.ipynb`                                        |
| Figure 4     | `compare_smac3.ipynb`                                  |
| Figure 5     | `old_coup_bound_plot.ipynb`                            |
| Figure 6     | `Utility_testing.ipynb`                                |
| Figure 7     | `COUP+/plotting_COUP+.ipynb` & `Utility_testing.ipynb` |
| Figure 8     | `melon_usk.ipynb`                                      |
| Figure 9     | `Heatmaps.ipynb`                                       |
| Figure 10    | `melon_usk.ipynb`                                      |    
| Figure 11    | `Distance_Casmo_SMAC.ipynb`                            |
| Table 1      | `compare_smac3.ipynb`                                  |
| Table 2      | `smac2_3_numbers.ipynb`                                |
| Table 3      | `old_coup_bounds_plot.ipynb`                           |
| Table 5      | `Heatmaps.ipynb`                                       |

[//]: # (|Name                                       |Contains                               |   )

[//]: # (|---                                        |---                                    |)



[//]: # (### Packages)

[//]: # (|Name                                   |Version                |   )

[//]: # (|---                                    |---                    |)

[//]: # ()
[//]: # (All of these packages can be installed by running the following command from the root of this project:)

[//]: # ()
[//]: # (```)

[//]: # (pip install -r requirements.txt)

[//]: # (```)



## Contact
