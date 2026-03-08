import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver
from utils import parse_u, u_to_str, ensure_directory, random_seed, safe_save, colors, lw, fs

from smac.runhistory.runhistory import RunHistory


def validate(state, instances, captime, savepath):
    incumbents = set(state['i_stars'])

    if os.path.exists(savepath):
        validation_utilities = pickle.load(open(savepath, 'rb'))
    else:
        validation_utilities = {}
    for ii, i in enumerate(incumbents):
        if i not in validation_utilities:
            config = state['configs'][i]['config'] # the configuration 
            utilities = []
            for j, instance in enumerate(instances):                
                t = solver.dorun(config, instance, random_seed(), captime)
                if t < captime: # the run completed 
                    utilities.append(u_fn(t, **u_params))
                else: # the run timed out
                    utilities.append(0)
                print(f"Validating config i={i} (number {ii + 1} of {len(incumbents)}) on instance number {j + 1} of {len(instances)}... cpu_time={t:.3f}, utility={u_fn(t, **u_params):.3f}")
            validation_utilities[i] = np.mean(utilities)
            safe_save(validation_utilities, savepath)
        else:
            print(f"Loading saved validation data for config i={i} (number {ii + 1} of {len(incumbents)})")

    return validation_utilities


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('runid', help="a string to help identify the run", nargs='?', default="runid", type=str)
    parser.add_argument('--seed', help="random seed", nargs='?', default=98, type=int)
    parser.add_argument('--runsolver', help="path to runsolver", default="../runsolver/runsolverx64/runsolver", type=str)
    parser.add_argument('--captime', help="captime to be used for validation runs", nargs='?', default=60, type=float)
    parser.add_argument('--max_num_instances', help="the maximum number of instances to validate", nargs='?', default=200, type=int)
    parser.add_argument('--overwrite', help="overwrite existing data for this run", action='store_true')

    args = parser.parse_args()

    print("Loading solver and instances ... ")

    validation_savepath = f"data/{args.runid}/validation_data.p"
    training_savepath = f"data/{args.runid}/training_data.p"

    if args.overwrite:
        print(f"Deleting validation data saved at {validation_savepath} ... ")
        if os.path.isfile(validation_savepath):
            os.remove(validation_savepath)

    run_args = pickle.load(open(f"data/{args.runid}/args.p", 'rb'))

    solver = Solver(run_args.solver, args.runsolver, seed=args.seed)

    instances = [os.path.join(run_args.instancedir, f) for f in os.listdir(run_args.instancedir) if f.endswith('.cnf')]
    instances.sort() # sort for reproducibility 
    np.random.seed(args.seed)
    np.random.shuffle(instances)

    instances = instances[:min(len(instances), args.max_num_instances)] # limit validation to at most max_num_instances instances

    u_str = u_to_str(run_args.u)
    u_fn, u_params = run_args.u

    print(f"\nValidating {args.runid} run with solver={run_args.solver} and instancedir={run_args.instancedir} and u={u_str} ...\n")
    
    print("Loading run state ...")
    if "coup" in args.runid:
        state = pickle.load(open(training_savepath, 'rb'))
    elif "smac" in args.runid:   
        runhistory = json.load(open(f"smac3_output/{args.runid}/0/runhistory.json", 'r'))
        state = pickle.load(open(training_savepath, 'rb'))
        incumbents = [1 if i is None else i for i in state['i_stars']] # replace Nones with default config
        state['i_stars'] = incumbents
        state['configs'] = {}
        for i in set(incumbents):
            state['configs'][i] = {}
            state['configs'][i]['config'] = runhistory['configs'][str(i)]
    else:
        print(f"runid={args.runid} not recognized...")
        exit()

    print("Validating ... ")
    validation_utilities = validate(state, instances, args.captime, validation_savepath)

    print(f"Finished validation for {args.runid}. Data saved to {validation_savepath}.")





