#!usr/bin/env python3

"""
Script for building csv from runs
"""

__author__ = "Skippybal"
__version__ = "0.1"

import sys
import glob
import time
import pandas as pd

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

STORAGE_LOC = "./storage/default7"


def main():

    all_data = []

    for filehandle in sorted(glob.glob(f"{STORAGE_LOC}/*.txt"), key=lambda x: int(x.split("/")[-1].split(".")[0])):
        # print(filehandle)
        data = pd.read_csv(filehandle, header=None)
        # print(data)
        # print(list(data.iloc[0]))
        # print([filehandle]+list(data.iloc[0]))
        # print([filehandle].extend(list(data.iloc[0])))
        all_data.append([filehandle]+list(data.iloc[0]))
    print(all_data)


    df = pd.DataFrame(all_data, columns=['File', 'Status', 'Runtime', 'Runlength', 'quality', 'seed'])

    # print dataframe.
    print(df)
    df.to_csv("test_data.csv")


    # print(len(glob.glob("instances/mip/data/SDPdMLPa-MIPVerify/*.lp")))
    #
    # all_files =sorted(glob.glob("instances/mip/data/SDPdMLPa-MIPVerify/*.lp"), key=lambda x: int(x.split("/")[-1].split(".")[0][4:]) )
    #
    # all_files = all_files#[:8]



    return 0




if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)