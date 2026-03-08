# Practical, Utilitarian Algorithm Configuration

This repo contains code for running the experiments associated with the paper Practical, Utilitarian Algorithm Configuration [https://arxiv.org/abs/2510.14683]. It requires `ACLib`, `runsolver` and `xgboost`. 

## Installation: 

Download [ACLib](https://bitbucket.org/mlindauer/aclib2/src/master/) into the repo's parent directory: 
```
git clone https://bitbucket.org/mlindauer/aclib2.git
```

Install `ACLib` requirements:
```
cd aclib2/
pip install -r requirements.txt
```

Download the `FACTORING`  example instance set:
```
wget http://data.aclib.net/sat_FACTORING.tar.gz
tar xvfz sat_FACTORING.tar.gz
```

The parameter space for each solver is described in the `solver.py` file. See the `ACLib` repo for other solvers and instance sets. 

Unzip `runsolver.zip` into the parent directory:
```
cd ../
unzip practical-coup/runsolver.zip
```

Install `xgboost` for the model:
```
pip install xgboost
```

The code for SMAC was taken from the SMAC3 [repo](https://github.com/automl/SMAC3) and altered slightly to save the incumbent configuration at each point in time. Navigate to the repo and unzip `smac`:
```
cd practical-coup
unzip smac.zip
```

## COUP vs SMAC Example Comparison

This default example is for configuring the `Spear-32_1.2.1` solver on the `FACTORING` instance set for the Pareto utility function, given by $u(t) = \frac{0.01}{t}$ if $t \ge 0.01$ and $u(t) = 1$ if $t < 0.01$. Other solvers, instance sets and utility functions can be passed at the command line using the `solver`, `instancedir` and `--u` arguments. The example runs for 120 CPU seconds. In practice, more time is needed. 

Run COUP:
```
python coup_runner.py ../aclib2/target_algorithms/sat/spear/Spear-32_1.2.1 ../aclib2/instances/sat/data/FACTORING/ --runid=coup_spear_factoring --max_cpu_time=120 
```

Run SMAC:
```
python smac_runner.py ../aclib2/target_algorithms/sat/spear/Spear-32_1.2.1 ../aclib2/instances/sat/data/FACTORING/ --runid=smac_spear_factoring --max_cpu_time=120 
```

Perform validation:
```
python validate.py coup_spear_factoring
python validate.py smac_spear_factoring
```

Plot the results:
```
python plot.py coup_spear_factoring smac_spear_factoring
```
The list of solvers used is `[...]`.

The list of instance sets used is `[...]`.

## Comparisson of COUP to Other Procedures with Guarantees 

This comparison is made in the original COUP repo: [https://github.com/drgrhm/coup]. In that repo, run 
```
python gap_experiment.py [minisat | cplex_rcw | cplex_region]
```

## SAT Competition Utility Function 

The SAT Competition data used can be found at https://www.cs.ubc.ca/~drgraham/datasets.html (original source: https://satcompetition.github.io/2023/downloads.html).

Download and unpack the data:
```
mkdir data
wget https://www.cs.ubc.ca/~drgraham/datasets/sc2023-detailed-results.zip
unzip sc2023-detailed-results.zip -d data
```

Generate the plots: 
```
python satcomp_experiment.py
```

