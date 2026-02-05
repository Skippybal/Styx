#!usr/bin/env python3

"""
Script to test calling the wrapper
"""

__author__ = "Skippybal"
__version__ = "0.1"

import sys

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import pcs_new, pcs

# from target_algorithms.bbob.branin.braninWrapper import runlength
from aclib2.target_algorithms.gurobi902.wrapper2 import PipelineWrapper


def main():
    with open('aclib2/target_algorithms/gurobi902/params_test.pcs', 'r') as fh:

        deserialized_conf = pcs_new.read(fh)

        # print(deserialized_conf)
    default = deserialized_conf.get_default_configuration()
    print(default)

    wrapped_runner = PipelineWrapper()
    instance_path = '/home/skippybal/Projects/Styx/aclib2/instances/mip/data/SDPdMLPa-MIPVerify/mip_29.lp'
    specifics = '0'
    cutoff = '9600.0'
    runlength = '2147483647'
    seed = '-1'
    start = ['--runsolver-path',
             '/home/skippybal/Projects/Styx/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver',
             instance_path, specifics, cutoff, runlength, seed]
    # start = ['--runsolver-path', '/home/skippybal/Projects/THESIS/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver', instance_path, '0', '9600.0', '2147483647', '-1']

    variabls = []
    print(dict(default))
    for variable, value in dict(default).items():
        variabls.append(f"-{variable}")
        variabls.append(f'{value}')

    # variabls.extend(["runsolvtarget_argser", "None"])

    all_args = start + variabls
    # print(variabls)
    # breakpoint()
    wrapped_runner.main(all_args)
    # wrapped_runner.main(["--threads", "1"])
    sys.stdout.write(
        "Result for SMAC: %s, %s, %s, %s, %s" % (wrapped_runner._ta_status, str(wrapped_runner._ta_runtime), str(wrapped_runner._ta_runlength),
                                                 str(wrapped_runner._ta_quality), str(wrapped_runner._seed)))
    if (len(wrapped_runner._ta_misc) > 0):
        sys.stdout.write(", %s" % (wrapped_runner._ta_misc))


    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)