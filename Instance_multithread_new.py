#!usr/bin/env python3

"""
Script for testing multiprocessing
"""

__author__ = "Skippybal"
__version__ = "0.1"

import sys
import glob
import time

import numpy as np
from pathlib import Path

import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import pcs_new, pcs


# from target_algorithms.bbob.branin.braninWrapper import runlength
from aclib2.target_algorithms.gurobi902.wrapper2 import PipelineWrapper

import multiprocessing as mp
from multiprocessing import Pool
from dotenv import load_dotenv
import os

STORAGE_LOC = "./storage/Instance_multithread_new_testing"

def process_configstring():
    # configstring = "-aggregate '1' -auto_aggfill_on '1' -auto_barcorrectors '1' -auto_barhomogeneous '1' -auto_barorder '1' -auto_bqpcuts_on '1' -auto_branchdir_on '1' -auto_cliquecuts_on '1' -auto_covercuts_on '1' -auto_crossover '1' -auto_cutaggpasses_on '1' -auto_cutpasses_on '1' -auto_cuts_on '1' -auto_degenmoves_on '1' -auto_disconnected_on '1' -auto_flowcovercuts_on '1' -auto_flowpathcuts_on '1' -auto_gomorypasses_on '0' -auto_gubcovercuts_on '0' -auto_impliedcuts_on '1' -auto_infproofcuts_on '1' -auto_issmethod_on '1' -auto_minrelnodes_on '1' -auto_mipsepcuts_on '1' -auto_mircuts_on '1' -auto_modkcuts_on '1' -auto_networkcuts_on '1' -auto_nodemethod_on '1' -auto_normadjust_on '1' -auto_predual_on '1' -auto_prepasses_on '1' -auto_presolve_on '1' -auto_presparsify_on '1' -auto_projimpliedcuts_on '1' -auto_pumppasses_on '1' -auto_quad_on '1' -auto_relaxliftcuts_on '1' -auto_rins_on '1' -auto_rltcuts_on '0' -auto_sifting_on '1' -auto_siftmethod_on '1' -auto_simplexpricing_on '0' -auto_startnodelimit '1' -auto_strongcgcuts_on '1' -auto_submipcuts_on '1' -auto_symmetry_on '0' -auto_varbranch_on '1' -auto_zerohalfcuts_on '1' -auto_zeroobjnodes_on '1' -crossoverbasis '0' -dualreductions '1' -gomorypasses '16' -gubcovercuts '1' -heuristics '0.049999999999999996' -improvestartgap '0.0' -improvestartnodes '2000000.0' -improvestarttime '1711941.2860724456' -infunbdinfo '0' -mipfocus '0' -partitionplace '15' -perturbvalue '2.0000000000000004E-4' -precrush '0' -predeprow '1' -rltcuts '1' -shut_off_mip_start_processing '0' -simplexpricing '0' -submipnodes '229' -symmetry '1' -threads '1'"
    configstring = "-aggregate '1' -auto_aggfill_on '1' -auto_barcorrectors '1' -auto_barhomogeneous '1' -auto_barorder '1' -auto_bqpcuts_on '1' -auto_branchdir_on '1' -auto_cliquecuts_on '1' -auto_covercuts_on '1' -auto_crossover '1' -auto_cutaggpasses_on '1' -auto_cutpasses_on '1' -auto_cuts_on '1' -auto_degenmoves_on '1' -auto_disconnected_on '1' -auto_flowcovercuts_on '1' -auto_flowpathcuts_on '1' -auto_gomorypasses_on '1' -auto_gubcovercuts_on '1' -auto_impliedcuts_on '1' -auto_infproofcuts_on '1' -auto_issmethod_on '1' -auto_minrelnodes_on '1' -auto_mipsepcuts_on '1' -auto_mircuts_on '1' -auto_modkcuts_on '1' -auto_networkcuts_on '1' -auto_nodemethod_on '1' -auto_normadjust_on '1' -auto_predual_on '1' -auto_prepasses_on '1' -auto_presolve_on '1' -auto_presparsify_on '1' -auto_projimpliedcuts_on '1' -auto_pumppasses_on '1' -auto_quad_on '1' -auto_relaxliftcuts_on '1' -auto_rins_on '1' -auto_rltcuts_on '1' -auto_sifting_on '1' -auto_siftmethod_on '1' -auto_simplexpricing_on '1' -auto_startnodelimit '1' -auto_strongcgcuts_on '1' -auto_submipcuts_on '1' -auto_symmetry_on '1' -auto_varbranch_on '1' -auto_zerohalfcuts_on '1' -auto_zeroobjnodes_on '1' -crossoverbasis '0' -dualreductions '1' -heuristics '0.049999999999999996' -improvestartgap '0.0' -improvestartnodes '2000000.0' -improvestarttime '2000000.0' -infunbdinfo '0' -mipfocus '0' -partitionplace '15' -perturbvalue '2.0000000000000004E-4' -precrush '0' -predeprow '-1' -shut_off_mip_start_processing '0' -submipnodes '500' -threads '1'"
    configstring = configstring.replace("'", "")
    args_list = configstring.split(" ")
    return args_list

def verify_instance(instance_loc: str, args_list, output_queue):

    wrapped_runner = PipelineWrapper()
    # instance_path = '/home/skippybal/Projects/THESIS/aclib2/instances/mip/data/SDPdMLPa-MIPVerify/mip_29.lp'
    instance_path = instance_loc
    specifics = '0'
    cutoff = '9600.0' #'5'#'9600.0'
    runlength = '2147483647'
    seed = '-1'
    start = ['--runsolver-path',
             #'/home/skippybal/Projects/Styx/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver',
             os.getenv("RUNSOLVER_LOC"),
             instance_path, specifics, cutoff, runlength, seed]
    # start = ['--runsolver-path', '/home/skippybal/Projects/THESIS/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver', instance_path, '0', '9600.0', '2147483647', '-1']

    # # sys.stdout.write(config)
    # variabls = []
    # # print(dict(config))
    # for variable, value in dict(config).items():
    # # for variable, value in config.items():
    #     variabls.append(f"-{variable}") #TODO: does this need to be doulbe dash?
    #     variabls.append(f'{value}')

    # variabls.extend(["runsolvtarget_argser", "None"])
    # print(sys.argv)

    # TODO: check sys.argv....

    all_args = start + args_list
    # print(all_args)
    # breakpoint()

    try:
        wrapped_runner.main(all_args)
    except Exception as e:
        print(f"Error found, {e}")
        print(f"Returning EXTERNAKILL with max captime")


    output_queue.put((wrapped_runner._ta_status, str(wrapped_runner._ta_runtime),
            str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality), str(wrapped_runner._seed), instance_loc))


def main():

    load_dotenv()

    with open('aclib2/target_algorithms/gurobi902/params_test.pcs', 'r') as fh:

        deserialized_conf = pcs_new.read(fh)

    default = deserialized_conf.get_default_configuration()
    print(default)

    Path(STORAGE_LOC).mkdir(parents=True, exist_ok=True)

    for filehandle in sorted(glob.glob("aclib2/instances/mip/data/SDPdMLPa-MIPVerify/*.lp"), key=lambda x: int(x.split("/")[-1].split(".")[0][4:]) ):
        print(filehandle)

    print(len(glob.glob("aclib2/instances/mip/data/SDPdMLPa-MIPVerify/*.lp")))

    all_files =sorted(glob.glob("aclib2/instances/mip/data/SDPdMLPa-MIPVerify/*.lp"), key=lambda x: int(x.split("/")[-1].split(".")[0][4:]) )

    all_files = all_files

    process_data_list = []
    max_processes = 10 #TODO: env variable for this???


    output_queue = mp.Queue()

    all_data = []

    args_list = process_configstring()

    instance_index = 0
    while instance_index < len(all_files):
        cleanup_done = False
        while not cleanup_done:
            cleanup_done = True
            for i, process_data in enumerate(process_data_list):
                if not process_data[1].is_alive():
                    print(f"finished {process_data[0]}")
                    p = process_data_list.pop(i)

                    new_result = output_queue.get()
                    all_data.append(new_result)

                    df = pd.DataFrame.from_records(all_data, columns=['Status', 'Runtime', 'Runlength', 'quality', 'seed', "File"])
                    df.to_csv(f"{STORAGE_LOC}/all_outputs.csv")

                    # breakpoint()
                    del p
                    cleanup_done = False
                    break

        if len(process_data_list) < max_processes:
            process = mp.Process(target=verify_instance, args=(all_files[instance_index], args_list, output_queue))
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
                all_data.append(new_result)

                df = pd.DataFrame.from_records(all_data,
                                               columns=['Status', 'Runtime', 'Runlength', 'quality', 'seed', "File"])
                df.to_csv(f"{STORAGE_LOC}/all_outputs.csv")

                del p
                cleanup_done = False
                break


    return 0




if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)