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
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import pcs_new, pcs
from dask.distributed import Client, progress
import concurrent.futures

# from target_algorithms.bbob.branin.braninWrapper import runlength
from aclib2.target_algorithms.gurobi902.wrapper2 import PipelineWrapper

import multiprocessing as mp
from multiprocessing import Pool

STORAGE_LOC = "./storage/default_test"

def verify_instance(instance_loc: str, config: ConfigurationSpace, output_queue):
    wrapped_runner = PipelineWrapper()
    # instance_path = '/home/skippybal/Projects/THESIS/aclib2/instances/mip/data/SDPdMLPa-MIPVerify/mip_29.lp'
    instance_path = instance_loc
    specifics = '0'
    cutoff = '5'#'9600.0'
    runlength = '2147483647'
    seed = '-1'
    start = ['--runsolver-path',
             '/home/skippybal/Projects/Styx/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver',
             instance_path, specifics, cutoff, runlength, seed]
    # start = ['--runsolver-path', '/home/skippybal/Projects/THESIS/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver', instance_path, '0', '9600.0', '2147483647', '-1']

    # sys.stdout.write(config)
    variabls = []
    # print(dict(config))
    for variable, value in dict(config).items():
    # for variable, value in config.items():
        variabls.append(f"-{variable}")
        variabls.append(f'{value}')

    # variabls.extend(["runsolvtarget_argser", "None"])

    all_args = start + variabls
    wrapped_runner.main(all_args)

    # sys.stdout.write(
    #     "Result for SMAC: %s, %s, %s, %s, %s" % (wrapped_runner._ta_status, str(wrapped_runner._ta_runtime),
    #                                              str(wrapped_runner._ta_runlength),
    #                                              str(wrapped_runner._ta_quality), str(wrapped_runner._seed)))
    # if (len(wrapped_runner._ta_misc) > 0):
    #     sys.stdout.write(", %s" % (wrapped_runner._ta_misc))

    # Path("../storage/default").mkdir(parents=True, exist_ok=True)
    with open(f"{STORAGE_LOC}/{instance_loc.split('/')[-1].split('.')[0][4:]}.txt", "w+") as f:
        # print(f.read())
        # f.write(f"{2}, {str(2)} \n")
        f.write(f"{wrapped_runner._ta_status}, {str(wrapped_runner._ta_runtime)}, {str(wrapped_runner._ta_runlength)},"
                f"{str(wrapped_runner._ta_quality)}, {str(wrapped_runner._seed)}")
    # return [1,1,1,1,1]
    output_queue.put((wrapped_runner._ta_status, str(wrapped_runner._ta_runtime),
            str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality), str(wrapped_runner._seed)))
    return [wrapped_runner._ta_status, str(wrapped_runner._ta_runtime),
            str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality), str(wrapped_runner._seed)]




def main():
    with open('aclib2/target_algorithms/gurobi902/params_test.pcs', 'r') as fh:

        deserialized_conf = pcs_new.read(fh)

        # print(deserialized_conf)
    default = deserialized_conf.get_default_configuration()
    print(default)

    Path(STORAGE_LOC).mkdir(parents=True, exist_ok=True)

    for filehandle in sorted(glob.glob("aclib2/instances/mip/data/SDPdMLPa-MIPVerify/*.lp"), key=lambda x: int(x.split("/")[-1].split(".")[0][4:]) ):
        print(filehandle)
        # with open(f"./storage/default/{filehandle.split('/')[-1].split('.')[0][4:]}.txt", "w+") as f:
        #     # print(f.read())
        #     f.write(
        #         f"{2}, {str(1)}")
        # open(f"./storage/default2/{filehandle.split('/')[-1].split('.')[0][4:]}.txt", "w+").close()

    # verify_instance(default)

    print(len(glob.glob("aclib2/instances/mip/data/SDPdMLPa-MIPVerify/*.lp")))
    # sorted(glob.glob("instances/mip/data/SDPdMLPa-MIPVerify/*.lp"), key=lambda x: int(x.split("/")[-1].split(".")[0][4:]) )
    # client = Client(threads_per_worker=1, n_workers=1)
    # client = Client()
    # print(client)
    # breakpoint()
    all_files =sorted(glob.glob("aclib2/instances/mip/data/SDPdMLPa-MIPVerify/*.lp"), key=lambda x: int(x.split("/")[-1].split(".")[0][4:]) )

    # futures = client.map(verify_instance, all_files, [default for i in range(len(all_files))]) #, scheduler="processes")
    # results = client.gather(futures)
    # print(len(results))
    # print(results)
    all_files = all_files#[:8]

    # with Pool(4) as p:
    # with Pool(4) as p:
    #     print(p.starmap(verify_instance, zip(all_files, [default for _ in range(len(all_files))] )))

    # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    #     full_futures = [executor.submit(verify_instance, file, default) for file in all_files ]
    #     concurrent.futures.wait(full_futures)
    # outputs = [future.result() for future in full_futures]
    # for i in outputs:
    #     print(outputs)

    # Other possible solution?
    # with Pool(3) as p:
    #     reslist = [p.apply_async(verify_instance, (file, default)) for file in all_files]
    #     for result in reslist:
    #         print(result.get())

    process_data_list = []
    max_processes = 4

    output_queue = mp.Queue()

    instance_index = 0
    while instance_index < len(all_files):
        cleanup_done = False
        while not cleanup_done:
            cleanup_done = True
            for i, process_data in enumerate(process_data_list):
                if not process_data[1].is_alive():
                    print(f"finished {process_data[0]}")
                    p = process_data_list.pop(i)
                    breakpoint()
                    del p
                    cleanup_done = False
                    break

        if len(process_data_list) < max_processes:
            process = mp.Process(target=verify_instance, args=(all_files[instance_index], default, output_queue))
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
                del p
                cleanup_done = False
                break


    return 0




if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)