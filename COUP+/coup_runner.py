'''
'''
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver
from coup import COUP
from utils import parse_u, u_to_str, ensure_directory
from dotenv import load_dotenv



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    ## experimental setup
    parser.add_argument("solver", help="path to solver")
    parser.add_argument("instancedir", help="path to instance directory")
    parser.add_argument('--runid', help="a string to help identify the run", nargs='?', default="runid", type=str)
    parser.add_argument('--u', help="utility function", nargs='?', default="u_pareto k0=.01 a=1", type=parse_u)
    parser.add_argument('--seed', help="random seed", nargs='?', default=98, type=int)
    parser.add_argument('--runsolver', help="path to runsolver", default="../runsolver/runsolverx64/runsolver", type=str)
    parser.add_argument('--overwrite', help="overwrite existing data for this run", action='store_true')

    ## coup parameters 
    parser.add_argument('--delta', help="failure probability", nargs='?', default=.01, type=float)
    parser.add_argument('--k0', help="initial captime", nargs='?', default=1, type=float)
    parser.add_argument('--n0', help="initial number of configs to test", nargs='?', default=50, type=int)    
    parser.add_argument('--max_wallclock_time', help="max wallclock time to spend running coup, seconds", nargs='?', default=1e10, type=float)
    parser.add_argument('--max_cpu_time', help="max cpu time to spend running coup, seconds", nargs='?', default=1e10, type=float)
    parser.add_argument('--savemod', help="how frequently to save state (iterations)", nargs='?', default=50, type=int)
    parser.add_argument('--model', help="use a model to predict performance and sample configs", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--usehoeffding', help="use hoeffding bounds instead of kl", action='store_true')
    parser.add_argument('--useucb', help="use UCB algorithm instead of LUCB", action='store_true')

    args = parser.parse_args()
    
    if args.k0 < 1:
        print(f"\nWARNING: initial captime is k0={args.k0} but must be int and >= 1. Setting k0=1\n")
        args.k0 = 1

    u_str = u_to_str(args.u)
    u_fn, u_params = args.u

    solver = Solver(args.solver, args.runsolver, seed=args.seed)

    # instances = [os.path.join(args.instancedir, f) for f in os.listdir(args.instancedir) if f.endswith('.cnf')]
    # TODO: fix this to use instance file
    # instances = [os.path.join(args.instancedir, f) for f in os.listdir(args.instancedir) if f.endswith('.lp')]
    load_dotenv()
    with open(os.getenv("TRAIN_INSTANCE_FILE")) as filehandle:
        instances = [line.rstrip() for line in filehandle]
    breakpoint()
    instances.sort() # sort for reproducibility 
    np.random.seed(args.seed)
    np.random.shuffle(instances)

    ensure_directory(f"data/{args.runid}")
    state_savepath = f"data/{args.runid}/training_data.p"

    if os.path.exists(state_savepath) and not args.overwrite:
        state = pickle.load(open(state_savepath, 'rb'))
        print(f"\nLoading saved state data found at {state_savepath}...\n")
    else:
        state = {}

    with open(f"data/{args.runid}/args.txt", 'w') as file: # save args used for this run 
        file.write(str(sys.argv)) # human readable 
    pickle.dump(args, open(f"data/{args.runid}/args.p", 'wb')) # pickle

    coup = COUP(solver, instances, lambda t: u_fn(t, **u_params), delta=args.delta, n0=args.n0, k0=args.k0, state=state, savepath=state_savepath, usemodel=args.model, usehoeffding=args.usehoeffding, useucb=args.useucb)

    print("Optimizing COUP ...")
    coup.optimize(max_cpu_time=args.max_cpu_time, max_wallclock_time=args.max_wallclock_time, savemod=args.savemod)
    
    guarantee = coup.guarantee()

    print(f"\nCOUP proved epsilon={guarantee['epsilon']}, gamma={guarantee['gamma']} for incumbent i_star={guarantee['i_star']} with configuration:")
    print(coup.get_configuration(guarantee['i_star']))










