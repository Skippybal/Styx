import ConfigSpace.hyperparameters.categorical
import numpy as np
from test_funcs.base import TestFunction

# Func2C and Func3C as appeared in CoCaBO
from ConfigSpace.read_and_write import pcs_new
import multiprocessing as mp
from ConfigSpace import ConfigurationSpace
from wrapper2 import PipelineWrapper
import glob
import time
import sys
from dotenv import load_dotenv, dotenv_values
import os


class MIPVerifyOracle(TestFunction):
    problem_type = 'mixed'

    def __init__(self, lamda=1e-6, normalize=False):
        super(MIPVerifyOracle, self).__init__(normalize)
        self.env_config = dotenv_values(".env")

        self.categorical_dims, self.continuous_dims, self.dim, self.config, self.lb, self.ub, self.index_choice_dict, self.int_constrained_dims, self.names = self._process_param_space()
        # categorical, continuous, len(categorical) + len(
        #     continuous), n_categories, cont_lb, cont_ub, cat_choices, int_disc

        # self.categorical_dims = np.arange(0, 50)
        # self.continuous_dims = np.array([50, 51, 52])
        # self.dim = len(self.continuous_dims) + len(self.categorical_dims)
        # self.n_vertices = 2 * np.ones(len(self.categorical_dims), dtype=int)
        # self.config = self.n_vertices
        # self.lamda = lamda
        # # specifies the range for the continuous variables
        # self.lb, self.ub = np.array([-1, -1, -1]), np.array([+1, +1, +1])

        # self._process_param_space()

        # load_dotenv()

        self.all_files = sorted(glob.glob(f"{self.env_config['INSTANCE_DIR']}/*.lp"),
                           key=lambda x: int(x.split("/")[-1].split(".")[0][4:]))

    def _process_param_space(self):
        # param_file_loc = os.getenv('PARAM_FILE_LOC')
        # breakpoint()
        param_file_loc = self.env_config['PARAM_FILE_LOC']
        # load_dotenv()
        # breakpoint()
        with open(param_file_loc, 'r') as fh:
            deserialized_conf = pcs_new.read(fh)
        deserialized_conf.seed(721)

        # TODO: it wants the first d_cat dims to be the categorical ones

        # continuous = []
        cont_names = []
        # categorical = []
        cat_names = []
        int_disc = []

        cont_ub = []
        cont_lb = []

        n_categories = []
        cat_choices = {}

        names = []
        int_names = []

        # print(dict(deserialized_conf))
        for index, (name, obj) in enumerate(dict(deserialized_conf).items()):
            # print(index, name, obj)
            # print(type(obj))
            # names.append(name)
            if type(obj) == ConfigSpace.hyperparameters.categorical.CategoricalHyperparameter:
                # print("cata")
                # print(obj.choices)
                # categorical.append(index)
                # print(obj)
                cat_choices[name] = obj.choices
                # print(obj.choices)
                n_categories.append(len(obj.choices))
                cat_names.append(name)

            elif type(obj) == ConfigSpace.hyperparameters.uniform_integer.UniformIntegerHyperparameter:
                # print("uniformint")
                # print(obj.range)

                # continuous.append(index)
                cont_lb.append(obj.lower)
                cont_ub.append(obj.upper)
                # int_disc.append(index)
                int_names.append(name)
                cont_names.append(name)

            elif type(obj) == ConfigSpace.hyperparameters.uniform_float.UniformFloatHyperparameter:
                # print("uniformFloat")
                # print(dir(obj))
                # print(obj.lower)
                # print(obj.upper)
                # print(obj.get_size())
                # continuous.append(index)
                cont_lb.append(obj.lower)
                cont_ub.append(obj.upper)
                cont_names.append(name)

            else:
                print(f"unknown: {obj}")
            # breakpoint()
        # print(continuous)
        # print(categorical)
        # print(int_disc)
        # print(cont_ub)
        # print(cont_lb)
        # print(n_categories)
        # print(cat_choices)
        all_names = cat_names + cont_names
        int_disc = [all_names.index(word) for word in int_names]
        # breakpoint()
        return (np.arange(0, len(cat_names)), np.arange(len(cat_names), len(cat_names) + len(cont_names)), len(cat_names) + len(cont_names),
                np.array(n_categories, dtype=int), np.array(cont_lb), np.array(cont_ub), cat_choices, np.array(int_disc), all_names)

    def verify_instance(self, instance_loc: str, config_array, output_queue):
        wrapped_runner = PipelineWrapper()

        instance_path = instance_loc
        specifics = '0'
        cutoff = str(self.env_config["ORACLE_CAPTIME"]) #'10'  # '9600.0' # TODO: fix this cutoff
        runlength = '2147483647'
        seed = '-1'
        start = ['--runsolver-path',
                 # os.getenv("RUNSOLVER_LOC"),
                 self.env_config["RUNSOLVER_LOC"],
                 instance_path, specifics, cutoff, runlength, seed,
                 '-threads', '1'] # TODO: threads seperate here becuase integration with casmopolitan

        variabls = []
        # print(dict(config))
        for inde, value in enumerate(config_array):
            varname = self.names[inde]
            # for variable, value in config.items():
            variabls.append(f"-{varname}") #TODO: double or single dash? dont think it matters call seems to be correcct
            if inde in self.categorical_dims:
                # TODO: check this
                variabls.append(f'{self.index_choice_dict[varname][int(value)]}')
                # print(inde)
                # variabls.append(f'{self.index_choice_dict[inde][int(value) - 1]}')
                # try:
                #     variabls.append(f'{self.index_choice_dict[varname][int(value)]}')
                # except IndexError as e:
                #     # print(inde)
                #     # print(self.index_choice_dict)
                #     # print(int(value))
                #     # print(self.categorical_dims)
                #     # print(self.config)
                #     print(self.names[inde])
                #     print(config_array)
                #     print(inde)
                #     print(value)
                #     print(self.index_choice_dict[inde])
                #     for ind, n_cats in zip(self.categorical_dims, self.config):
                #         print(f"{ind}: {n_cats}")
                #     sys.exit()
            elif inde in self.int_constrained_dims:
                variabls.append(f'{int(value)}')
            else:
                variabls.append(f'{value}')

        # variabls.extend(["runsolvtarget_argser", "None"])
        # print(sys.argv)
        sys.argv = sys.argv[:1]
        all_args = start + variabls

        wrapped_runner.main(all_args)
        sys.argv = sys.argv[:1]

        if wrapped_runner._ta_status in ["CRACHED"]:
            wrapped_runner._ta_runtime = 10 # cutoff time

        output_queue.put((wrapped_runner._ta_status, float(wrapped_runner._ta_runtime),
                          str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality),
                          str(wrapped_runner._seed)))


    def _verify_instances_config(self, single_config):
        def log_laplace_single(t, k_0, alpha=1):
            if t < k_0:
                return 1 - 0.5 * (t / k_0) ** alpha
            else:
                return 0.5 * (k_0 / t) ** alpha

        all_utility = []

        process_data_list = []
        max_processes = 1

        output_queue = mp.Queue()

        instance_index = 0
        while instance_index < len(self.all_files):
            cleanup_done = False
            while not cleanup_done:
                cleanup_done = True
                for i, process_data in enumerate(process_data_list):
                    if not process_data[1].is_alive():
                        print(f"finished {process_data[0]}")
                        p = process_data_list.pop(i)
                        breakpoint()

                        new_result = output_queue.get()

                        # TODO: this captime thiny fix.... make consistent with the one above in verify_instance..
                        all_utility.append(log_laplace_single(new_result[0], int(self.env_config["K0_UTILITY"])))



                        del p
                        cleanup_done = False
                        break

            if len(process_data_list) < max_processes:
                process = mp.Process(target=self.verify_instance, args=(self.all_files[instance_index], single_config, output_queue))
                process.start()
                process_data_list.append([instance_index, process])
                print(f"starting {instance_index}: {self.all_files[instance_index]}")
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
                    all_utility.append(log_laplace_single(new_result[0], int(self.env_config["K0_UTILITY"])))

                    del p
                    cleanup_done = False
                    break

        return -np.mean(all_utility)

    def compute(self, X, normalize=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # To make sure there is no cheating, round the discrete variables before calling the function
        X[:, self.categorical_dims] = np.round(X[:, self.categorical_dims])

        X[:, self.int_constrained_dims] = np.round(X[:, self.int_constrained_dims])

        #TODO: round the int dims as well

        results = []
        for config in X:
            print(config)
            print("---------------------")
            # breakpoint()
            results.append(self._verify_instances_config(config))

        return np.array(results)

    # TODO: the sample normalize is to get a normaliztion I think, but we probably shouldnt, just build it into the utility function


if __name__ == '__main__':
    # f = Func3C()
    # print(f.sample_normalize(10))

    f = MIPVerifyOracle()
    f._process_param_space()
