import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver
from utils import parse_u, u_to_str, ensure_directory, safe_save

from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    ## experimental setup
    parser.add_argument("solver", help="path to solver")
    parser.add_argument("instancedir", help="path to instance directory")
    parser.add_argument('--runid', help="a string to help identify the run", nargs='?', default="runid", type=str)
    parser.add_argument('--u', help="utility function", nargs='?', default="u_pareto k0=.01 a=1", type=parse_u)
    parser.add_argument('--seed', help="random seed", nargs='?', default=98, type=int)
    parser.add_argument('--runsolver', help="path to runsolver", default="../runsolver/runsolverx64/runsolver", type=str)

    parser.add_argument("--instancefeatures", help="path to file describing instance features", nargs='?', default=None)
    parser.add_argument('--max_wallclock_time', help="wallclock time limit for smac, seconds", nargs='?', default=1e10, type=float)
    parser.add_argument('--max_cpu_time', help="cpu time limit for smac, seconds", nargs='?', default=1e10, type=float)
    parser.add_argument('--max_solver_time', help="cpu time limit for each execution of the solver", nargs='?', default=60, type=float)
    parser.add_argument('--logginglevel', help="logging verbosity", nargs='?', default=0, type=float)
    parser.add_argument('--overwrite', help="overwrite existing data for this runid", action='store_true')
    parser.add_argument('--load_u_from_file', help="load the utility function from a given file path", nargs='?', default=None, type=str)

    args = parser.parse_args()
    
    ensure_directory(f"data/{args.runid}")

    solver = Solver(args.solver, args.runsolver, seed=args.seed)

    instances = [os.path.join(args.instancedir, f) for f in os.listdir(args.instancedir) if f.endswith('.cnf')]
    instances.sort() # sort for reproducibility 
    np.random.seed(args.seed)
    np.random.shuffle(instances)

    if args.load_u_from_file:
        print(f"Loading utility function from {args.load_u_from_file}, ignoring --u argument")
        u_data = pickle.load(open(args.load_u_from_file, 'rb'))
        instance_name = args.instancedir.split('/')[-2]
        u = parse_u(u_data[(solver.name, instance_name)])
        u_str = u_to_str(u)
        u_fn, u_params = u
    else:
        u_str = u_to_str(args.u)
        u_fn, u_params = args.u

    print(f"Optimizing utility function {u_str}")

    print(f"Running SMAC with max cutoff time {args.max_solver_time}")

    training_savepath = f"data/{args.runid}/training_data.p"

    with open(f"data/{args.runid}/args.txt", 'w') as file: # save args used for this run 
        file.write(str(sys.argv)) # human readable 
    pickle.dump(args, open(f"data/{args.runid}/args.p", 'wb')) # pickle

    if args.instancefeatures is None:
        instance_features = dict([(inst, [i]) for i, inst in enumerate(instances)])
    else:
        df = pd.read_csv(args.instancefeatures, sep=',')
        df.set_index('INSTANCE_ID', inplace=True)
        instance_features = {}
        for inst, feat in df.iterrows():
            if "../aclib2/" + inst in instances:
                instance_features["../aclib2/" + inst] = list(feat)


    if os.path.exists(training_savepath) and not args.overwrite:
        state = pickle.load(open(training_savepath, 'rb'))
        observation_times_cpu = state['observation_times_cpu']
        incumbents = state['i_stars']
        total_cpu_time = observation_times_cpu[-1]
    else:
        observation_times_cpu = []
        incumbents = []
        total_cpu_time = 0
    

    # The function passed to smac for doing runs. To be minimized. 
    # We keep track of the cpu time consumed here using runsolver for consistency with COUP. 
    def train(config, instance, seed=None): 
        global total_cpu_time
        global smac
        
        # get the current incumbent from smac's runhistory
        if len(smac.runhistory) > 1:
            trialkey = list(smac.runhistory)[-2]
            trialvalue = smac.runhistory[trialkey]
            incumbent = trialvalue.additional_info['incumbent']
        else:
            incumbent = None
        incumbents.append(incumbent)

        # and the current total_cpu_time
        observation_times_cpu.append(total_cpu_time)

        # save the current incumbent and total_cpu_time
        safe_save({'observation_times_cpu': observation_times_cpu, 'i_stars': incumbents}, training_savepath)

        # do the next run
        t = solver.dorun(config, instance, seed, args.max_solver_time)
        total_cpu_time += t

        print(f"\nSMAC: total_cpu_time={total_cpu_time:.3f} of max_cpu_time={args.max_cpu_time}, \n")

        if total_cpu_time > args.max_cpu_time:
            raise KeyboardInterrupt

        return 1 - u_fn(t, **u_params)



    scenario = Scenario(solver.configspace, instances=instances, instance_features=instance_features, use_default_config=True, name=args.runid, walltime_limit=args.max_wallclock_time, n_trials=1e10)
    smac = ACFacade(scenario, train, overwrite=args.overwrite, logging_level=args.logginglevel)
    
    try:
        smac_incumbent = smac.optimize()
    except KeyboardInterrupt:
        print(f"SMAC terminated at total_cpu_time={total_cpu_time}")

    
    print(f"\nSMAC returned incumbent with configuration:")
    print(smac.runhistory.ids_config[incumbents[-1]])





