#!usr/bin/env python3

"""
"""

__author__ = "Skippybal"
__version__ = "0.1"

import math
import random
import sys
import csv
import glob
import time
import pandas as pd

import numpy as np
from pathlib import Path
from ConfigSpace import ConfigurationSpace
import ConfigSpace as CS
from ConfigSpace.read_and_write import pcs_new, pcs
from dask.distributed import Client, progress
import concurrent.futures

# from target_algorithms.bbob.branin.braninWrapper import runlength
from aclib2.target_algorithms.gurobi902.wrapper2 import PipelineWrapper

import multiprocessing as mp
from multiprocessing import Pool

STORAGE_LOC = "./storage/COUP_multi/run1"

class COUP:
    def __init__(self, train_paths, test_paths, config_space):
        self.train = train_paths
        self.test = test_paths
        self.config_space = config_space
        self.all_data = {}
        self.configs = {} # TODO: This could also be a list? probably...

        with open(self.train, 'r') as file:
            self.train_files = [line.strip() for line in file]

        random.seed(721)
        random.shuffle(self.train_files)

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

    def write_configs(self):
        Path(STORAGE_LOC).mkdir(parents=True, exist_ok=True)
        # print(CS.get_active_hyperparameters(self.config_space))

        # print(self.config_space.items())
        # print(self.config_space.keys())
        # print(list(self.config_space.keys()))

        # print(self.configs[0])
        # print(self.configs[0].get_array())

        with open(f"{STORAGE_LOC}/configs.csv", "w+") as filehandle:
            # w = csv.DictWriter(filehandle, ["id"] + list(self.configs[0].keys()))
            w = csv.DictWriter(filehandle, ["id"] + list(self.config_space.keys()))
            w.writeheader()
            # w.writerow(def_dict)
            for key,val in sorted(self.configs.items()):
                row = {'id': key}
                row.update(val)
                w.writerow(row)

        return 0

    def write_csv(self, config_id):
        Path(STORAGE_LOC).mkdir(parents=True, exist_ok=True)
        fields = ['instance_id', 'file', 'status', 'runtime', 'runlength', 'quality', 'seed', 'captime']
        with open(f"{STORAGE_LOC}/{config_id}.csv", "w+") as filehandle:
            csv_out = csv.writer(filehandle)
            csv_out.writerow(fields)
            for key, val in self.all_data[config_id].items():
                csv_out.writerow([key, self.train_files[key], *val])
            # for row in data:
            #     csv_out.writerow(row)

        return 0

    def verify_instance(self, output_queue, config_id, instance_id, cutoff, specifics="0", runlength="2147483647", seed="-1", runsolver_path='/home/skippybal/Projects/Styx/aclib2/configurators/smac/example_scenarios/spear-generic-wrapper/runsolver'):
        # TODO: could also give flag: captime_increase, and then only do this loop as the instance id should exist

        # Because captime is now in the tuple, we can keep seeing which captime was used to generate the run for later testing
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

        instance_path = self.train_files[instance_id]

        start = ['--runsolver-path',
                 runsolver_path,
                 instance_path, specifics, str(cutoff), runlength, seed]

        # sys.stdout.write(config)
        variabls = []
        # print(dict(config))
        # TODO: store as dict so no need to return
        # for variable, value in self.configs[instance_id].items():
        for variable, value in dict(self.configs[config_id]).items():
            # for variable, value in config.items():
            variabls.append(f"-{variable}")
            variabls.append(f'{value}')


        all_args = start + variabls
        print(all_args)


        # store sys.argv as the wrapper extends them
        #TODO: why did this work for the instance verification? (Because only used default)
        tmp = list(sys.argv) # This is ref otherwise
        wrapped_runner.main(all_args)
        sys.argv = tmp
        # print(sys.argv)
        # breakpoint()

        # # Path("../storage/default").mkdir(parents=True, exist_ok=True)
        # with open(f"{STORAGE_LOC}/{instance_loc.split('/')[-1].split('.config_id')[0][4:]}.txt", "w+") as f:
        #     # print(f.read())
        #     # f.write(f"{2}, {str(2)} \n")
        #     f.write(
        #         f"{wrapped_runner._ta_status}, {str(wrapped_runner._ta_runtime)}, {str(wrapped_runner._ta_runlength)},"
        #         f"{str(wrapped_runner._ta_quality)}, {str(wrapped_runner._seed)}")
        #TODO: instance_id or instance loc
        #TODO: store in outer loop not here

        if wrapped_runner._ta_status=="CRASHED":
            ... # TODO make this captime?

        output_queue.put((config_id, instance_id, wrapped_runner._ta_status, float(wrapped_runner._ta_runtime),
                str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality), str(wrapped_runner._seed), cutoff))
        # return (config_id, instance_id, wrapped_runner._ta_status, float(wrapped_runner._ta_runtime),
        #         str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality), str(wrapped_runner._seed), cutoff)


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

            # print(U_hat)
            U_hat = np.concatenate((U_hat, np.zeros(n_p - ns[-1]))) #Add zeros
            # print(U_hat)
            # breakpoint()

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

                self.all_data[i] = {}

                if i == 0:
                    self.configs[i] = self.config_space.get_default_configuration()
                else:
                    self.configs[i] = self.config_space.sample_configuration()

            self.write_configs()
            breakpoint()

            ns.append(n_p)

            i_prime = np.argmax(UCB)
            i_star = np.argmax(LCB)
            epsilon_star = 1

            # Starting processes
            process_data_list = []
            max_processes = 4 #TODO: .env

            # TODO: close this somewhere, also make class attribute?
            output_queue = mp.Queue()

            r = 0
            while UCB[i_prime] - LCB[i_star] >= epsilon_fn(p):

                cleanup_done = False
                while not cleanup_done:
                    cleanup_done = True
                    for i, process_data in enumerate(process_data_list):
                        if not process_data.is_alive():
                            # print(f"finished {process_data[0]}")
                            proc = process_data_list.pop(i)
                            # breakpoint()
                            del proc
                            cleanup_done = False


                            new_result = output_queue.get()

                            # TODO: implement
                            # self.process_results()

                            config_id, instance_id, ta_status, runtime, ta_runlength, ta_quality, seed, used_captime = new_result

                            self.all_data[config_id][instance_id] = (ta_status, runtime, ta_runlength, ta_quality, seed,
                                                                     used_captime)
                            # config_id

                            F_hat[config_id] = ((m[config_id] - 1) * F_hat[config_id] + (1 if runtime < k[config_id] else 0)) / m[config_id]
                            U_hat[config_id] = ((m[config_id] - 1) * U_hat[config_id] + utility(runtime)) / m[config_id]

                            self.write_csv(config_id)

                            # alpha_i = self.alpha_p(p, n_p, m[config_id], k[config_id], delta)
                            # instance_id+1 to keep it the same as what goes into the process? but what if another finishes first?
                            #TODO: maybe not like this but m+1 only once finished.... but problem when picking the same different number of times...
                            # maybe this is best as it assumes the others will finishs at some point, but ucb is still not correct, so update not completely correct
                            alpha_i = self.alpha_p(p, n_p, instance_id+1, k[config_id], delta)
                            # alpha_i = self.alpha_p(p, n_p, m[i]+1, k[config_id], delta)
                            UCB[config_id] = min(U_hat[config_id] + (1 - utility(k[config_id])) * alpha_i, UCB[config_id])
                            LCB[config_id] = max(U_hat[config_id] - alpha_i - utility(k[config_id]) * (1 - F_hat[config_id]), LCB[config_id])

                            # TODO: check? is this tiebreaking based on LCB first and the maximizing U_hat?
                            if improved_tie_breaking:
                                i_star = self.choose_max(LCB, U_hat)
                            else:
                                i_star = np.argmax(LCB)

                            i_prime = np.argmax(UCB)
                            epsilon_star = UCB[i_prime] - LCB[i_star]


                            break

                if len(process_data_list) < max_processes:

                    i = np.argmax(UCB)

                    m[i] += 1
                    if m[i] >= m_max:
                        print(
                            "\nWARNING: coup reached m_max={} samples at round r={} in phase p={}. returning.\n".format(
                                m_max, r, p))
                        # TODO: fix this
                        self.update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=0,
                                           total_times=0)
                        print(self.coup_message(p, r, n_p, i_star, epsilon_star, UCB, LCB, m,
                                                k) + " RAN OUT OF INSTANCES")
                        return out

                    # TODO: check if this alpha_p matters much as its used/recalculated later, but there after multi procesiign m[i] might have changed, or even worse k[i]
                    # TODO: only k_i can change in the original code.. maybe give
                    alpha_i = self.alpha_p(p, n_p, m[i], k[i], delta)


                    # TODO: this is to soon.... maybe? should be later
                    if doubling_condition == "old":
                        dubcond = 2 * alpha_i <= utility(k[i]) * (1 - F_hat[i])
                    elif doubling_condition == "new":
                        dubcond = 2 * (1 - utility(k[i])) * alpha_i <= utility(k[i]) * (1 - F_hat[i] + alpha_i)

                    if dubcond:

                        #TODO: this needs a seperate function that waits for an empty queue and redoes everything
                        # for a specific config
                        k[i] = 2 * k[i]

                        runtimes = []
                        for j in range(m[i]):
                            config_id, instance_id, ta_status, runtime, ta_runlength, ta_quality, seed, used_captime = self.verify_instance(
                                i, m[i] - 1, k[i])
                            time.sleep(5)

                            self.all_data[config_id][instance_id] = (ta_status, runtime, ta_runlength, ta_quality, seed,
                                                                     used_captime)
                            runtimes.append(runtime)

                        # TODO: here we need an extra wait to make sure it runs these things and waits for results....
                        # Need to have empty queue so we know all of them are for this config....
                        # seperate loop as if we dont we wont know if the results from the queue are for
                        # how about making this a seperate function? self.increase captime

                        F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m[i]
                        U_hat[i] = sum(utility(t) for t in runtimes) / m[i]


                    else:

                        #TODO: check if this gets wonky with multiprocess
                        process = mp.Process(target=self.verify_instance,
                                             args=(output_queue, i, m[i] - 1, k[i]))
                        process.start()
                        process_data_list.append(process)
                    # process_data_list.append([instance_index, process])
                    # print(f"starting {instance_index}: {all_files[instance_index]}")
                    # instance_index += 1
                else:
                    time.sleep(0.1)

                # if len(process_data_list) >= max_processes:
                #     time.sleep(10)
                #     continue




                # TODO: this should change when using many cores, maybe sort by UCB and take top N?
                # i = np.argmax(UCB)
                #
                #
                # # here we should insert a process waiting/listening loop
                #
                # m[i] += 1
                # if m[i] >= m_max:
                #     print("\nWARNING: coup reached m_max={} samples at round r={} in phase p={}. returning.\n".format(
                #         m_max, r, p))
                #     # TODO: fix this
                #     self.update_output(out, i_star=i_star, epsilon_star=epsilon_star, total_time=0,
                #                   total_times=0)
                #     print(self.coup_message(p, r, n_p, i_star, epsilon_star, UCB, LCB, m, k) + " RAN OUT OF INSTANCES")
                #     return out
                #
                # alpha_i = self.alpha_p(p, n_p, m[i], k[i], delta)

                # if doubling_condition == "old":
                #     dubcond = 2 * alpha_i <= utility(k[i]) * (1 - F_hat[i])
                # elif doubling_condition == "new":
                #     dubcond = 2 * (1 - utility(k[i])) * alpha_i <= utility(k[i]) * (1 - F_hat[i] + alpha_i)


                # if dubcond:
                #     k[i] = 2 * k[i]
                #
                #     runtimes = []
                #     for j in range(m[i]):
                #         config_id, instance_id, ta_status, runtime, ta_runlength, ta_quality, seed, used_captime = self.verify_instance(i, m[i] - 1, k[i])
                #         time.sleep(5)
                #
                #         self.all_data[config_id][instance_id] = (ta_status, runtime, ta_runlength, ta_quality, seed, used_captime)
                #         runtimes.append(runtime)
                #
                #     # TODO: here we need an extra wait to make sure it runs these things and waits for results....
                #     # Need to have empty queue so we know all of them are for this config....
                #     # seperate loop as if we dont we wont know if the results from the queue are for
                #     # how about making this a seperate function? self.increase captime
                #
                #     F_hat[i] = sum([1 if t < k[i] else 0 for t in runtimes]) / m[i]
                #     U_hat[i] = sum(utility(t) for t in runtimes) / m[i]
                # else: # otherwise, just run the next instance

                    # runtime = env.run(i, m[i] - 1, k[i])
                    # config_id, instance_id, ta_status, runtime, ta_runlength, ta_quality, seed, used_captime = self.verify_instance(i, m[i] - 1, k[i])
                #     # time.sleep(5)
                #     self.all_data[config_id][instance_id] = (ta_status, runtime, ta_runlength, ta_quality, seed, used_captime)
                #
                #     F_hat[i] = ((m[i] - 1) * F_hat[i] + (1 if runtime < k[i] else 0)) / m[i]
                #     U_hat[i] = ((m[i] - 1) * U_hat[i] + utility(runtime)) / m[i]
                #
                # self.write_csv(i)
                #
                # alpha_i = self.alpha_p(p, n_p, m[i], k[i], delta)
                # UCB[i] = min(U_hat[i] + (1 - utility(k[i])) * alpha_i, UCB[i])
                # LCB[i] = max(U_hat[i] - alpha_i - utility(k[i]) * (1 - F_hat[i]), LCB[i])
                #
                # #TODO: check? is this tiebreaking based on LCB first and the maximizing U_hat?
                # if improved_tie_breaking:
                #     i_star = self.choose_max(LCB, U_hat)
                # else:
                #     i_star = np.argmax(LCB)
                #
                # i_prime = np.argmax(UCB)
                # epsilon_star = UCB[i_prime] - LCB[i_star]


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

    @staticmethod
    def log_laplace_single(t, k_0, alpha=1):
        if t < k_0:
            return 1 - 0.5 * (t / k_0) ** alpha
        else:
            return 0.5 * (k_0 / t) ** alpha


def epsilon_fn(p):
    return math.exp(- (p / 6))

def gamma_fn(p):
    return math.exp(- p / 3)

def main():
    with open('aclib2/target_algorithms/gurobi902/params_test.pcs', 'r') as fh:
        deserialized_conf = pcs_new.read(fh)

    deserialized_conf.seed(721)

    runner = COUP("./aclib2/instances/mip/sets/SDPdMLPa-MIPVerify/training.txt",
                  "./aclib2/instances/mip/sets/SDPdMLPa-MIPVerify/test.txt",
                  deserialized_conf)
    k_0 = 5
    delta = 0.05
    runner.run(utility=lambda t: COUP.log_laplace_single(t, k_0=k_0, alpha=1), delta=delta,
               epsilon_fn=epsilon_fn, gamma_fn=gamma_fn, k0=k_0,
               # max_phases=args.numphases, n_max=env.num_configs, m_max=env.num_instances,
               doubling_condition="new", improved_tie_breaking=True)
    return 0




if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)