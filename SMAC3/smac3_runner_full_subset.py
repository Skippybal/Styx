'''
'''

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

# from solver import Solver
from utils import parse_u, u_to_str, ensure_directory, safe_save

from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario

from dotenv import load_dotenv
from ConfigSpace.read_and_write import pcs_new
import ConfigSpace
from wrapper2 import PipelineWrapper
# from CASMOPOLITAN.wrapper2 import  PipelineWrapper
from copy import copy
from smac.intensifier.intensifier import Intensifier

import multiprocessing as mp
import time
import glob


def verify_instance(instance_loc: str, config: ConfigSpace, output_queue):

    wrapped_runner = PipelineWrapper()

    instance_path = instance_loc
    specifics = '0'
    cutoff = str(os.getenv("ORACLE_CAPTIME")) #str(cutoff)
    runlength = '2147483647'
    seed = '-1'
    start = ['--runsolver-path',
             # os.getenv("RUNSOLVER_LOC"),
             os.getenv("RUNSOLVER_LOC"),
             instance_path, specifics, cutoff, runlength, seed,
             '-threads', '1' # TODO: check this for all methods
             ]

    variabls = []

    for variable, value in dict(config).items():
        variabls.append(f"-{variable}")
        variabls.append(f'{value}')

    tmp = copy(sys.argv)
    sys.argv = sys.argv[:1]
    all_args = start + variabls

    try:
        wrapped_runner.main(all_args)
    except UnicodeDecodeError as e:
        print(f"Error found, {e}")
        print(f"Returning EXTERNALKILL with max captime")
        wrapped_runner._ta_runtime = int(int(os.getenv(
            "ORACLE_CAPTIME")))  # TODO: this doesnt really matter with utility func as util will be 0, but does matter in smac....

    sys.argv = tmp  # ys.argv[:1]

    if wrapped_runner._ta_status in ["CRACHED"]:
        wrapped_runner._ta_runtime = int(os.getenv("ORACLE_CAPTIME"))  # 10 # cutoff time

    output_queue.put((wrapped_runner._ta_status, float(wrapped_runner._ta_runtime),
                      str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality),
                      str(wrapped_runner._seed), instance_loc))
    # return (wrapped_runner._ta_status, float(wrapped_runner._ta_runtime),
    #         str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality),
    #         str(wrapped_runner._seed))

GLOBAL_CONFIG_NUMBER = 0

def _verify_instances_config(single_config, args, u_fn, u_params, all_files):
    global GLOBAL_CONFIG_NUMBER
    GLOBAL_CONFIG_NUMBER += 1

    all_utility = []
    all_runtime = []

    process_data_list = []
    max_processes = int(os.getenv("MAX_PROCESSES")) #1

    output_queue = mp.Queue()

    # self.verify_instance(self.all_files[0], single_config, output_queue)
    # breakpoint()

    instance_index = 0
    while instance_index < len(all_files):
        cleanup_done = False
        while not cleanup_done:
            cleanup_done = True
            for i, process_data in enumerate(process_data_list):
                if not process_data[1].is_alive():
                    print(f"finished {process_data[0]}")
                    p = process_data_list.pop(i)
                    # breakpoint()

                    new_result = output_queue.get()

                    # TODO: this captime thiny fix.... make consistent with the one above in verify_instance..
                    # all_utility.append(log_laplace_single(new_result[1], int(self.env_config["K0_UTILITY"])))
                    # all_utility.append(u_geometric(new_result[1], int(os.getenv("K0_UTILITY")), int(os.getenv("K1_UTILITY"))))
                    all_utility.append(u_fn(new_result[1], **u_params))

                    all_runtime.append(new_result[1])

                    with open(f"data/{args.runid}/{int(os.getenv('SMAC_SEED'))}/all_instances.csv", "a") as filehandle:
                        filehandle.write(f"{GLOBAL_CONFIG_NUMBER},{new_result[-1]},{new_result[1]},{u_fn(new_result[1], **u_params)}\n")



                    del p
                    cleanup_done = False
                    break

        if len(process_data_list) < max_processes:
            process = mp.Process(target=verify_instance, args=(all_files[instance_index], single_config, output_queue))
            process.start()
            process_data_list.append([instance_index, process])
            print(f"starting {instance_index}: {all_files[instance_index]}")
            instance_index += 1
        else:
            time.sleep(0.1)

    while process_data_list:
        for i, process_data in enumerate(process_data_list):
            if not process_data[1].is_alive():
                print(f"finished {process_data[0]}")
                p = process_data_list.pop(i)

                new_result = output_queue.get()

                # TODO: this captime thiny fix.... make consistent with the one above in verify_instance..
                # all_utility.append(log_laplace_single(new_result[1], int(self.env_config["K0_UTILITY"])))
                # all_utility.append(u_geometric(new_result[1], int(os.getenv("K0_UTILITY")), int(os.getenv("K1_UTILITY"))))
                all_utility.append(u_fn(new_result[1], **u_params))

                all_runtime.append(new_result[1])

                with open(f"data/{args.runid}/{int(os.getenv('SMAC_SEED'))}/all_instances.csv", "a") as filehandle:
                    filehandle.write(f"{GLOBAL_CONFIG_NUMBER},{new_result[-1]},{new_result[1]},{u_fn(new_result[1], **u_params)}\n")


                del p
                cleanup_done = False
                break
    # breakpoint()
    # all_runtimes.append(np.mean(all_runtime))

    if os.getenv("OBJECTIVE") == "PAR":
        return np.mean(all_runtime)
    else:
        return 1 - np.mean(all_utility) # u_fn(t, **u_params)

    return np.mean(all_utility)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()

    ## experimental setup
    parser.add_argument("solver", help="path to solver")
    parser.add_argument("instancedir", help="path to instance directory")
    parser.add_argument('--runid', help="a string to help identify the run", nargs='?', default="runid", type=str)
    parser.add_argument('--u', help="utility function", nargs='?', default="u_pareto k0=.01 a=1", type=parse_u)
    parser.add_argument('--seed', help="random seed", nargs='?', default=98, type=int)
    # parser.add_argument('--runsolver', help="path to runsolver", default="../runsolver/runsolverx64/runsolver",
    #                     type=str)

    parser.add_argument("--instancefeatures", help="path to file describing instance features", nargs='?', default=None)
    parser.add_argument('--max_wallclock_time', help="wallclock time limit for smac, seconds", nargs='?', default=1e10,
                        type=float)
    parser.add_argument('--max_cpu_time', help="cpu time limit for smac, seconds", nargs='?', default=1e10, type=float)
    parser.add_argument('--max_solver_time', help="cpu time limit for each execution of the solver", nargs='?',
                        default=60, type=float)
    parser.add_argument('--logginglevel', help="logging verbosity", nargs='?', default=0, type=float)
    parser.add_argument('--overwrite', help="overwrite existing data for this runid", action='store_true')
    # parser.add_argument('--load_u_from_file', help="load the utility function from a given file path", nargs='?',
    #                     default=None, type=str)

    args = parser.parse_args()

    if not os.path.exists(f"data/{args.runid}"):
        os.makedirs(f"data/{args.runid}")

    os.makedirs(f'data/{args.runid}/{int(os.getenv("SMAC_SEED"))}', exist_ok=True)
    with open(f"data/{args.runid}/{int(os.getenv('SMAC_SEED'))}/all_instances.csv", "w") as file:
        file.write(f"Config_ID, Instance,Runtime,Utility\n")

    with open(os.getenv("TRAIN_INSTANCE_FILE")) as filehandle:
        instances = [line.rstrip() for line in filehandle]

    instances.sort()  # sort for reproducibility
    all_instances = sorted(glob.glob(f"{os.getenv('INSTANCE_DIR')}/*.lp"),
                           key=lambda x: int(x.split("/")[-1].split(".")[0][4:]))
    # print(all_instances)
    instances = instances[:1] # only take first to make smac thing we have 1 instance

    np.random.seed(args.seed)
    np.random.shuffle(instances)

    u_str = u_to_str(args.u) # TODO
    u_fn, u_params = args.u

    print(f"Optimizing utility function {u_str}")

    print(f"Running SMAC with max cutoff time {args.max_solver_time}")

    training_savepath = f"data/{args.runid}/training_data.p"

    with open(f"data/{args.runid}/args.txt", 'w') as file:  # save args used for this run
        file.write(str(sys.argv))  # human readable
    pickle.dump(args, open(f"data/{args.runid}/args.p", 'wb'))  # pickle

    if args.instancefeatures is None:
        instance_features = dict([(inst, [i]) for i, inst in enumerate(instances)])

    # global total_cpu_time
    global smac

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

        # ta_status, runtime, ta_runlength, ta_quality, seed = verify_instance(instance, int(os.getenv("ORACLE_CAPTIME")), config,
        #                                                                           output_queue=None, seed=seed)
        #
        # t = runtime
        #
        # if os.getenv("OBJECTIVE") == "PAR":
        #     return t
        # else:
        #     return 1 - u_fn(t, **u_params)
        return _verify_instances_config(config, args, u_fn, u_params, all_instances)



    # TODO: so does the params_cosmo.pcs file have threads = 1? or does it need it seperatly (nope threads seperatly

    param_path = os.getenv("PARAM_FILE_LOC")
    with open(param_path, 'r') as fh:
        deserialized_conf = pcs_new.read(fh)

    scenario = Scenario(deserialized_conf, instances=instances, instance_features=instance_features,
                        use_default_config=True, name=args.runid, walltime_limit=args.max_wallclock_time, n_trials=1e10,
                        seed=int(os.getenv("SMAC_SEED")))
    # scenario.get

    intensifier = Intensifier(
        scenario=scenario,
        max_config_calls=1,
        max_incumbents=10,
    )

    smac = ACFacade(scenario, train, overwrite=args.overwrite, logging_level=args.logginglevel,
                    intensifier=intensifier)


    try:
        smac_incumbent = smac.optimize()
    except KeyboardInterrupt:
        print(f"SMAC terminated at total_cpu_time={total_cpu_time}")

    print(f"\nSMAC returned incumbent with configuration:")
    print(smac.runhistory.ids_config[incumbents[-1]])

    return 0


if __name__ == "__main__":
    exitcode = main()
    sys.exit(exitcode)





