#!usr/bin/env python3

"""
"""

__author__ = "Skippybal"
__version__ = "0.1"

import math
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

class COUP:
    def __init__(self, train_paths, test_paths, config_space):
        self.train = train_paths
        self.test = test_paths
        self.config_space = config_space
        self.all_data = {}
        self.configs = {} # TODO: This could also be a list? probably...

    @staticmethod
    def choose_max(main_array, secondary_array):
        choice_array = np.ones(main_array.shape) * -1
        choice_array[main_array == main_array.max()] = secondary_array[main_array == main_array.max()]
        return np.flatnonzero(choice_array == choice_array.max())[0]
        # return np.random.choice(np.flatnonzero(choice_array == choice_array.max()))

    @staticmethod
    def alpha_p(p, n, m, k, delta):
        return math.sqrt(math.log(36 * p ** 2 * n * m ** 2 * (math.log2(k) + 1) ** 2 / delta) / 2 / m)

    @staticmethod
    def coup_message(p, r, n_p, i_star, epsilon_star, UCB, LCB, m, k):
        return f"coup: phase p={p}. r={r}. n_p={n_p} configs sampled. i_star={i_star:5}, epsilon_star={epsilon_star:.4f}, ucb=[{np.min(UCB):.4f}, {np.max(UCB):.4f}], lcb=[{np.min(LCB):.4f}, {np.max(LCB):.4f}], m=[{min(m.values())}, {max(m.values())}], k=[{min(k.values())}, {max(k.values())}]"

    def verify_instance(self, config_id, instance_id, cutoff, specifics="0", runlength="2147483647", seed="-1", runsolver_path='/home/skippybal/Projects/Styx/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver'):
        # TODO: could also give flag: captime_increase, and then only do this loop as the instance id should exist
        if instance_id in self.all_data[config_id]:
            if self.all_data[config_id][instance_id][0] != "TIMEOUT":
                return self.all_data[config_id][instance_id][0]

        wrapped_runner = PipelineWrapper()
        # instance_path = '/home/skippybal/Projects/THESIS/aclib2/instances/mip/data/SDPdMLPa-MIPVerify/mip_29.lp'
        # instance_path = instance_loc
        # specifics = '0'
        # cutoff = '9600.0'
        # runlength = '2147483647'
        # seed = '-1'
        start = ['--runsolver-path',
                 runsolver_path,
                 instance_path, specifics, cutoff, runlength, seed]

        # sys.stdout.write(config)
        variabls = []
        # print(dict(config))
        for variable, value in dict(self.configs[instance_id]).items():
            # for variable, value in config.items():
            variabls.append(f"-{variable}")
            variabls.append(f'{value}')


        all_args = start + variabls
        wrapped_runner.main(all_args)

        # Path("../storage/default").mkdir(parents=True, exist_ok=True)
        with open(f"{STORAGE_LOC}/{instance_loc.split('/')[-1].split('.')[0][4:]}.txt", "w+") as f:
            # print(f.read())
            # f.write(f"{2}, {str(2)} \n")
            f.write(
                f"{wrapped_runner._ta_status}, {str(wrapped_runner._ta_runtime)}, {str(wrapped_runner._ta_runlength)},"
                f"{str(wrapped_runner._ta_quality)}, {str(wrapped_runner._seed)}")
        #TODO: instance_id or instance loc
        #TODO: store in outer loop not here
        return (config_id, instance_id, wrapped_runner._ta_status, str(wrapped_runner._ta_runtime),
                str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality), str(wrapped_runner._seed))


    @staticmethod
    def update_output(out, **kwargs):
        out['i_stars'].append(kwargs['i_star'])
        out['epsilon_stars'].append(kwargs['epsilon_star'])
        out['total_time'].append(kwargs['total_time'])
        out['total_times'].append(kwargs['total_time'])

    def run(self, utility, delta, epsilon_fn, gamma_fn,
            k0=1, max_phases=float('inf'), n_max=float('inf'),
            m_max=float('inf'), save_mod=500, print_mod=10000,
            doubling_condition="new", improved_tie_breaking=False):

        F_hat = {} # Fractions of runs completed?
        U_hat = []
        m = {}
        k = {}
        ns = [0]  # number of configs per phase

        out = {'phase': [],
               'i_stars': [],
               'epsilon_stars': [],
               'total_time': [],
               'total_times': []
        }

        p = 1
        while p <= max_phases:

            # Decides how many configurations should be in this phase
            n_p = math.ceil(math.log(math.pi ** 2 * p ** 2 / 3 / delta) / gamma_fn(p))
            if n_p >= n_max:
                print(
                    "\nWARNING: coup needs n_p={} >= n_max={} configurations for phase p={}. returning phase {} results.\n".format(
                        n_p, n_max, p, p - 1))
                return out

            UCB = np.ones(n_p)
            LCB = np.zeros(n_p)

            U_Hat = np.concatenate((U_hat, np.zeros(n_p - ns[-1]))) #Add zeros

            # TODO: figure out why we do this this way
            # alpha_p is confidence width as defined in COUP paper but I dont know exactely what it do
            for i in range(ns[-1]):  # updates for existing configs
                if m[i] >= 1:  # if we've run i before
                    UCB[i] = min(U_hat[i] + (1 - utility(k[i])) * self.alpha_p(p, n_p, m[i], k[i], delta), UCB[i])
                    LCB[i] = max(U_hat[i] - self.alpha_p(p, n_p, m[i], k[i], delta) - utility(k[i]) * (1 - F_hat[i]), LCB[i])

            for i in range(ns[-1], n_p):  # initializations for new configs
                F_hat[i] = 0
                m[i] = 0
                k[i] = k0
                # TODO: sample new configs here

            ns.append(n_p)

            i_prime = np.argmax(UCB)
            i_star = np.argmax(LCB)
            epsilon_star = 1

            r = 0
            while UCB[i_prime] - LCB[i_star] >= epsilon_fn(p):

                # TODO: this should change when using many cores, maybe sort by UCB and take top N?
                i = np.argmax(UCB)


                # here we should insert a process waiting/listening loop

                m[i] += 1
                if m[i] >= m_max:
                    print("\nWARNING: coup reached m_max={} samples at round r={} in phase p={}. returning.\n".format(
                        m_max, r, p))
                    # TODO: fix this
                    self.update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=0,
                                  total_times=0)
                    print(self.coup_message(p, r, n_p, i_star, epsilon_star, UCB, LCB, m, k) + " RAN OUT OF INSTANCES")
                    return out

                alpha_i = self.alpha_p(p, n_p, m[i], k[i], delta)

                if doubling_condition == "old":
                    dubcond = 2 * alpha_i <= utility(k[i]) * (1 - F_hat[i])
                elif doubling_condition == "new":
                    dubcond = 2 * (1 - utility(k[i])) * alpha_i <= utility(k[i]) * (1 - F_hat[i] + alpha_i)


                # TODO: after this it sould write to csv... but need to look into doulbess, or just write self.all_data to csv?
                if dubcond:
                    k[i] = 2 * k[i]
                    #TODO: here we need to make sure we only rerun the ones that have t>k

                    # runtimes = [env.run(i, j, k[i]) for j in range(m[i])]
                    runtimes = []
                    for j in range(m[i]):
                        config_id, instance_id, ta_status, runtime, ta_runlength, ta_quality, seed = self.verify_instance(i, m[i] - 1, k[i])
                        #TODO: here store to all data
                        # TODO: should this tuple also contain captime?
                        self.all_data[config_id][instance_id] = (ta_status, runtime, ta_runlength, ta_quality, seed)
                        runtimes.append(runtime)

                    F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m[i]
                    U_hat[i] = sum(utility(t) for t in runtimes) / m[i]
                else: # otherwise, just run the next instance

                    # runtime = env.run(i, m[i] - 1, k[i])
                    config_id, instance_id, ta_status, runtime, ta_runlength, ta_quality, seed = self.verify_instance(i, m[i] - 1, k[i])
                    self.all_data[config_id][instance_id] = (ta_status, runtime, ta_runlength, ta_quality, seed)

                    F_hat[i] = ((m[i] - 1) * F_hat[i] + (1 if runtime < k[i] else 0)) / m[i]
                    U_hat[i] = ((m[i] - 1) * U_hat[i] + utility(runtime)) / m[i]



                alpha_i = self.alpha_p(p, n_p, m[i], k[i], delta)
                UCB[i] = min(U_hat[i] + (1 - utility(k[i])) * alpha_i, UCB[i])
                LCB[i] = max(U_hat[i] - alpha_i - utility(k[i]) * (1 - F_hat[i]), LCB[i])

                #TODO: check? is this tiebreaking based on LCB first and the maximizing U_hat?
                if improved_tie_breaking:
                    i_star = self.choose_max(LCB, U_hat)
                else:
                    i_star = np.argmax(LCB)

                i_prime = np.argmax(UCB)
                epsilon_star = UCB[i_prime] - LCB[i_star]


                if r % save_mod == 0:
                    self.update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=0, total_times=0)

                if r % print_mod == 0:
                    print(self.coup_message(p, r, n_p, i_star, epsilon_star, UCB, LCB, m, k))
                r += 1

                # TODO: store entire object? then we also dont lose UCB and LCB... maybe a good idea?

            print(self.coup_message(p, r, n_p, i_star, epsilon_star, UCB, LCB, m, k) + ". PHASE {} COMPLETE.".format(p))

            self.update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=0,
                          total_times=0)
            out['phase'].append({
                'num_configs': n_p,
                'i_star': i_star,
                'epsilon': epsilon_star,
                'total_time': 0,#env.total_time / day_in_s, # Total time (in days?)
                'total_times': 0#env.total_times, # Time in seconds
            })

            p += 1

        return out

            # TODO, store each runs tuntime so we can do analysis later
            # so storing to text files, but we should make a text file that has each config and their ID
            # then under that ID we store the txt files, and we also store this coup objecct so we have everything saved

            # If runtime already below captime, don't rerun, otherise retry
            # also make counter for how many times we hit captime
            # and make a sampling thingy
            # how do ewe make sure no duplicatas? if we have predifined list
            # it could work, but what if not enough...
            # also once we start modeliing that won't work

            # Barplot over the things for the configs and the bounds?

            # Also dealing with if we have no more instances to verify....

            # TODO: over runtime plotting, so as in coup the env has total_time/s
            # So plot incumbent over CPU time?






def main():

    return 0




if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)