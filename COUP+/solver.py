import math
import subprocess
import numpy as np
from copy import copy, deepcopy
from scipy.stats import truncnorm

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, InCondition, AndConjunction
from ConfigSpace.read_and_write import pcs_new
import ConfigSpace
from dotenv import load_dotenv
import os
import sys
from wrapper2 import PipelineWrapper


class Solver():
    """ Class to hold the parameters of different solvers used"""
    def __init__(self, solver_path, runsolver, seed=None):
        self._path = solver_path
        self._runsolver = runsolver
        self._name = solver_path.split('/')[-1]
        load_dotenv()

        inf = 1e8 # the largest parameter vaule to use
        min_float_value = 1e-8 # the smallest float value to use

        if self._name == 'Spear-32_1.2.1':
            self._param_dict = {
                "--sp-var-dec-heur": {'type': Integer, 'lower': 0, 'upper': 19, 'default': 0, 'log': False},
                "--sp-learned-clause-sort-heur": {'type': Integer, 'lower': 0, 'upper': 19, 'default': 0, 'log': False},
                "--sp-orig-clause-sort-heur": {'type': Integer, 'lower': 0, 'upper': 19, 'default': 0, 'log': False},
                "--sp-res-order-heur": {'type': Integer, 'lower': 0, 'upper': 19, 'default': 0, 'log': False},
                "--sp-clause-del-heur": {'type': Integer, 'lower': 0, 'upper': 2, 'default': 2, 'log': False},
                "--sp-phase-dec-heur": {'type': Integer, 'lower': 0, 'upper': 6, 'default': 5, 'log': False},
                "--sp-resolution": {'type': Integer, 'lower': 0, 'upper': 2, 'default': 1, 'log': False},
                "--sp-variable-decay": {'type': Float, 'lower': 1, 'upper': 2, 'default': 1.4, 'log': True},
                "--sp-clause-decay": {'type': Float, 'lower': 1, 'upper': 2, 'default': 1.4, 'log': True},
                "--sp-restart-inc": {'type': Float, 'lower': 1.1, 'upper': 1.9, 'default': 1.5, 'log': False},
                "--sp-learned-size-factor": {'type': Float, 'lower': 0.1, 'upper': 1.6, 'default': 0.4, 'log': True},
                "--sp-learned-clauses-inc": {'type': Float, 'lower': 1.1, 'upper': 1.5, 'default': 1.3, 'log': False},
                "--sp-clause-activity-inc": {'type': Float, 'lower': 0.5, 'upper': 1.5, 'default': 1, 'log': False},
                "--sp-var-activity-inc": {'type': Float, 'lower': 0.5, 'upper': 1.5, 'default': 1, 'log': False},
                "--sp-rand-phase-dec-freq": {'type': Categorical, 'set': [0, 0.0001, 0.001, 0.005, 0.01, 0.05], 'default': 0.001},
                "--sp-rand-var-dec-freq": {'type': Categorical, 'set': [0, 0.0001, 0.001, 0.005, 0.01, 0.05], 'default': 0.001},
                "--sp-rand-var-dec-scaling": {'type': Float, 'lower': 0.3, 'upper': 1.1, 'default': 1, 'log': False},
                "--sp-rand-phase-scaling": {'type': Float, 'lower': 0.3, 'upper': 1.1, 'default': 1, 'log': False},
                "--sp-max-res-lit-inc": {'type': Float, 'lower': .25, 'upper': 4, 'default': 1, 'log': False},
                "--sp-first-restart": {'type': Integer, 'lower': 25, 'upper': 3200, 'default': 100, 'log': True},
                "--sp-res-cutoff-cls": {'type': Integer, 'lower': 2, 'upper': 20, 'default': 8, 'log': True},
                "--sp-res-cutoff-lits": {'type': Integer, 'lower': 100, 'upper': 1600, 'default': 400, 'log': True},
                "--sp-max-res-runs": {'type': Integer, 'lower': 1, 'upper': 32, 'default': 4, 'log': True},
                "--sp-update-dec-queue": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False},
                "--sp-use-pure-literal-rule": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False},
                "--sp-clause-inversion": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False}
            }
            self._parameters = self.construct_parameters()
            self._configspace = ConfigurationSpace(seed=seed)
            self._configspace.add_hyperparameters(self._parameters)
            self._conditions = [
                InCondition(self._configspace["--sp-rand-phase-dec-freq"], self._configspace["--sp-phase-dec-heur"], [0,1,3,4,5,6]),
                InCondition(self._configspace["--sp-rand-var-dec-scaling"], self._configspace["--sp-rand-var-dec-freq"], [0.0001, 0.001, 0.005, 0.01, 0.05]),
                InCondition(self._configspace["--sp-rand-phase-scaling"], self._configspace["--sp-rand-phase-dec-freq"], [0.0001, 0.001, 0.005, 0.01, 0.05]),
                InCondition(self._configspace["--sp-clause-inversion"], self._configspace["--sp-learned-clause-sort-heur"], [19]),
                InCondition(self._configspace["--sp-res-order-heur"], self._configspace["--sp-resolution"], [1, 2]),
                InCondition(self._configspace["--sp-max-res-lit-inc"], self._configspace["--sp-resolution"], [1, 2]),
                InCondition(self._configspace["--sp-res-cutoff-cls"], self._configspace["--sp-resolution"], [1, 2]),
                InCondition(self._configspace["--sp-res-cutoff-lits"], self._configspace["--sp-resolution"], [1, 2]),
                InCondition(self._configspace["--sp-max-res-runs"], self._configspace["--sp-resolution"], [1, 2]),
            ]
            self._configspace.add_condition(self._conditions)
            self._callstring = self._runsolver + " --{4}-limit={3} --delay=0 " + self._path + " {0} --time --dimacs={1} --seed={2}"
            
        elif self._name == 'cryptominisat':
            self._param_dict = {
                "--gluehist": {'type': Integer, 'lower': 80, 'upper': 300, 'default': 100, 'log': True},
                "--blkrest": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False},
                "--burst": {'type': Categorical, 'set': [100,300,1000,2000], 'default': 300},
                "--clbtwsimp": {'type': Integer, 'lower': 1, 'upper': 5, 'default': 2, 'log': False}, # should be log=True but that doesnt seem right for predictions...
                "--blkrestlen": {'type': Integer, 'lower': 2000, 'upper': 10000, 'default': 5000, 'log': True},
                "--blkrestmultip": {'type': Float, 'lower': 1.1, 'upper': 2, 'default': 1.4, 'log': False},
                "--ltclean": {'type': Float, 'lower': 0.3, 'upper': 0.7, 'default': 0.5, 'log': False},
                "--cleanconflmult": {'type': Integer, 'lower': 0, 'upper': 2, 'default': 1, 'log': False},
                "--lockuip": {'type': Integer, 'lower': 50, 'upper': 1000, 'default': 500, 'log': True}, # should be log=False but that doesnt seem right for predictions...
                "--locktop": {'type': Integer, 'lower': 0, 'upper': 2000, 'default': 0, 'log': False}, 
                "--perfmult": {'type': Float, 'lower': 0.00001, 'upper': 0.8, 'default': 0.00001, 'log': True},
                "--startclean": {'type': Integer, 'lower': 100, 'upper': 50000, 'default': 10000, 'log': True}, 
                "--incclean": {'type': Float, 'lower': 1.02, 'upper': 1.8, 'default': 1.1, 'log': False},
                "--freq": {'type': Float, 'lower': 0, 'upper': 0.05, 'default': 0, 'log': False},
                "--dompickf": {'type': Integer, 'lower': 100, 'upper': 10000, 'default': 400, 'log': True}, 
                "--morebump": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False}, 
                "--flippolf": {'type': Categorical, 'set': [0, 400], 'default': 0},
                "--calcpolar1st": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False}, 
                "--calcpolarall": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False}, 
                "--moreminimcache": {'type': Integer, 'lower': 20, 'upper': 600, 'default': 200, 'log': True}, 
                "--moreminimbin": {'type': Integer, 'lower': 20, 'upper': 5000, 'default': 100, 'log': True}, 
                "--probemaxm": {'type': Integer, 'lower': 800, 'upper': 4000, 'default': 1600, 'log': False}, 
                "--cachesize": {'type': Integer, 'lower': 1024, 'upper': 4096, 'default': 2048, 'log': True}, 
                "--calcreach": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False}, 
                "--cachecutoff": {'type': Integer, 'lower': 600, 'upper': 5000, 'default': 2000, 'log': True}, 
                "--presimp": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 0, 'log': False}, 
                "--elimcoststrategy": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 0, 'log': False}, 
                "--bva2lit": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False}, 
                "--eratio": {'type': Float, 'lower': 0.05, 'upper': 1, 'default': 0.12, 'log': True},
                "--occredmax": {'type': Integer, 'lower': 100, 'upper': 400, 'default': 200, 'log': True}, 
                "--sccperc": {'type': Float, 'lower': 0.001, 'upper': 0.5, 'default': 0.02, 'log': True},
                "--viviflongmaxm": {'type': Integer, 'lower': 5, 'upper': 80, 'default': 20, 'log': True}, 
                "--viviffastmaxm": {'type': Integer, 'lower': 100, 'upper': 1000, 'default': 400, 'log': True}, 
                "--restart": {'type': Categorical, 'set': ["geom", "agility", "glue", "glueagility"], 'default': "glue"},
                "--clean": {'type': Categorical, 'set': ["size", "glue", "activity", "prconf", "confdep"], 'default': "prconf"},
                "--updateglue": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 1, 'log': False}, 
            }
            self._parameters = self.construct_parameters()
            self._configspace = ConfigurationSpace(seed=seed)
            self._configspace.add_hyperparameters(self._parameters)
            self._conditions = [
                InCondition(self._configspace["--blkrestlen"], self._configspace["--blkrest"], [1]),
                InCondition(self._configspace["--blkrestmultip"], self._configspace["--blkrest"], [1]),
            ]
            self._configspace.add_condition(self._conditions)
            self._callstring = self._runsolver + " --{4}-limit={3} --delay=0 " + self._path + " {0} --random={2} --input={1}"

        elif self._name == 'lingeling':
            self._param_dict = {
                "plain": {'type': Categorical, 'set': [0, 1], 'default': 0},
                "locs": {'type': Integer, 'lower': -1, 'upper': 2147483647, 'default': 0, 'log': False},
                "locsclim": {'type': Integer, 'lower': 0, 'upper': 8388607, 'default': 1000000, 'log': False},
                "locsrtc": {'type': Categorical, 'set': [0, 1], 'default': 0},
                "locswait": {'type': Categorical, 'set': [0, 1, 2], 'default': 2},
                "decompose": {'type': Categorical, 'set': [0, 1], 'default': 1},
                "elim": {'type': Categorical, 'set': [0, 1], 'default': 1},
                "elmrtc": {'type': Categorical, 'set': [0, 1], 'default': 0},
                "scincinc": {'type': Integer, 'lower': 0, 'upper': 10000, 'default': 200, 'log': False},
                "phase": {'type': Categorical, 'set': [-1, 0, 1], 'default': 0},
                "redlinit": {'type': Integer, 'lower': 0, 'upper': 100000000, 'default': 4000, 'log': False},
                "reduce": {'type': Categorical, 'set': [0, 1, 2, 3, 4], 'default': 2},
                "redfixed": {'type': Categorical, 'set': [0, 1], 'default': 0},
                "acts": {'type': Categorical, 'set': [0, 1, 2], 'default': 2},
                "agile": {'type': Categorical, 'set': [0, 1, 2], 'default': 1},
                "block": {'type': Categorical, 'set': [0, 1], 'default': 1},
                "boost": {'type': Categorical, 'set': [0, 1], 'default': 1},
                "cce": {'type': Categorical, 'set': [0, 1, 2, 3], 'default': 3},
                "gauss": {'type': Categorical, 'set': [0, 1], 'default': 1},
                "minimize": {'type': Categorical, 'set': [0, 1, 2], 'default': 2},
                "otfs": {'type': Categorical, 'set': [0, 1], 'default': 1},
                "probe": {'type': Categorical, 'set': [0, 1], 'default': 1},
                "restart": {'type': Categorical, 'set': [0, 1, 2, 3], 'default': 2},
                "restartinit": {'type': Integer, 'lower': 0, 'upper': 512, 'default': 0, 'log': False},
                "restartint": {'type': Integer, 'lower': 1, 'upper': 1024, 'default': 5, 'log': False},
                "restartintscale": {'type': Categorical, 'set': [-1, 0, 1], 'default': 0},
                "score": {'type': Categorical, 'set': [0, 1, 2, 3, 4, 5, 6], 'default': 5},
            }
            self._parameters = self.construct_parameters()
            self._configspace = ConfigurationSpace(seed=seed)
            self._configspace.add_hyperparameters(self._parameters)
            self._callstring = self._runsolver + " --{4}-limit={3} --delay=0 " + self._path + " {0} --seed={2} {1}"

        elif self._name == 'glucose_static':
            self._param_dict = {
                "-luby": {'type': Categorical, 'set': ["-luby", "-no-luby"], 'default': "-no-luby"},
                "-gr": {'type': Categorical, 'set': ["-gr", "-no-gr"], 'default': "-gr"},
                "-rinc": {'type': Float, 'lower': 1.1, 'upper': 4, 'default': 2, 'log': False},
                "-cla-decay": {'type': Float, 'lower': 0.9, 'upper': 0.99999, 'default': 0.999, 'log': True},
                "-max-var-decay": {'type': Float, 'lower': 0.9, 'upper': 0.99999, 'default': 0.95, 'log': False},
                "-var-decay": {'type': Float, 'lower': 0.6, 'upper': 0.99999, 'default': 0.8, 'log': False},
                "-phase-restart": {'type': Integer, 'lower': 0, 'upper': 1, 'default': 0, 'log': False},
                "-phase-saving": {'type': Integer, 'lower': 0, 'upper': 2, 'default': 2, 'log': False},
                "-ccmin-mode": {'type': Integer, 'lower': 0, 'upper': 2, 'default': 2, 'log': False},
                "-minLBDMinimizingClause": {'type': Integer, 'lower': 3, 'upper': 20, 'default': 6, 'log': False},
                "-minSizeMinimizingClause": {'type': Integer, 'lower': 3, 'upper': 100, 'default': 30, 'log': False},
                "-chanseok": {'type': Categorical, 'set': ["-chanseok", "-no-chanseok"], 'default': "-no-chanseok"},
                "-specialIncReduceDB": {'type': Integer, 'lower': 10, 'upper': 100000, 'default': 1000, 'log': True},
                "-minLBDFrozenClause": {'type': Integer, 'lower': 10, 'upper': 1000, 'default': 30, 'log': True},
                "-luby-factor": {'type': Integer, 'lower': 1, 'upper': 1024, 'default': 100, 'log': True},
                "-co": {'type': Integer, 'lower': 2, 'upper': 16, 'default': 5, 'log': False},
                "-firstReduceDB": {'type': Integer, 'lower': 100, 'upper': 10000, 'default': 2000, 'log': True},
                "-incReduceDB": {'type': Integer, 'lower': 100, 'upper': 10000, 'default': 300, 'log': True},
                "-R": {'type': Float, 'lower': 1, 'upper': 5, 'default': 1.4, 'log': False},
                "-K": {'type': Float, 'lower': 0.5, 'upper': 0.99999, 'default': 0.8, 'log': False},
                "-szTrailQueue": {'type': Integer, 'lower': 100, 'upper': 100000, 'default': 5000, 'log': True},
                "-szLBDQueue": {'type': Integer, 'lower': 10, 'upper': 1024, 'default': 50, 'log': True},
                "-asymm": {'type': Categorical, 'set': ["-asymm", "-no-asymm"], 'default': "-no-asymm"},
                "-elim": {'type': Categorical, 'set': ["-elim", "-no-elim"], 'default': "-elim"},
                "-simp-gc-frac": {'type': Float, 'lower': 0.1, 'upper': 0.9, 'default': 0.5, 'log': False},
                "-grow": {'type': Integer, 'lower': 0, 'upper': 10, 'default': 0, 'log': False},
                "-cl-lim": {'type': Integer, 'lower': 1, 'upper': 128, 'default': 20, 'log': False},
                "-sub-lim": {'type': Integer, 'lower': 100, 'upper': 10000, 'default': 1000, 'log': False},
                "-update-act-trick": {'type': Categorical, 'set': ["-update-act-trick", "-no-update-act-trick"], 'default': "-update-act-trick"},
                "-forceunsat": {'type': Categorical, 'set': ["-forceunsat", "-no-forceunsat"], 'default': "-forceunsat"},
                "-fix-phas-rest": {'type': Categorical, 'set': ["-fix-phas-rest", "-no-fix-phas-rest"], 'default': "-no-fix-phas-rest"},
                "-rnd-init": {'type': Categorical, 'set': ["-rnd-init", "-no-rnd-init"], 'default': "-no-rnd-init"},
                "-ratio-rem-clauses": {'type': Float, 'lower': 1.1, 'upper': 10, 'default': 2, 'log': False},
                "-step-inc-var-decay": {'type': Float, 'lower': 0.000001, 'upper': 0.1, 'default': 0.01, 'log': False},
                "-lower-block-restart": {'type': Float, 'lower': 100, 'upper': 100000, 'default': 10000, 'log': False},
                "-good-cl-thresh": {'type': Float, 'lower': 1, 'upper': 30, 'default': 3, 'log': False},
                "-freq-inc-var-decay": {'type': Float, 'lower': 1, 'upper': 1000000, 'default': 5000, 'log': False},
            }
            self._parameters = self.construct_parameters()
            self._configspace = ConfigurationSpace(seed=seed)
            self._configspace.add_hyperparameters(self._parameters)
            self._conditions = [
                InCondition(self._configspace["-luby-factor"], self._configspace["-luby"], ["-luby"]),                
                InCondition(self._configspace["-co"], self._configspace["-chanseok"], ["-chanseok"]),
            ]
            self._configspace.add_condition(self._conditions)
            self._callstring = self._runsolver + " --{4}-limit={3} --delay=0 " + self._path + " {0} -rnd-seed={2} -no-adapt {1}"
        elif self._name == "MipVerify":
            param_path = os.getenv("PARAM_FILE_LOC")
            with open(param_path, 'r') as fh:
                deserialized_conf = pcs_new.read(fh)

            self._configspace = deserialized_conf
            self._configspace.seed(seed)
            self._param_dict = dict()

            for name, param in dict(self._configspace).items():

                match type(param):
                    # TODO: the dash is probably not necessary...
                    case ConfigSpace.hyperparameters.uniform_float.UniformFloatHyperparameter:
                        type_thing = Float
                        self._param_dict[f"{name}"] = {'type': type_thing, 'lower': param.lower, 'upper': param.upper,
                                                        'default': param.default_value, 'log': param.log}
                    case ConfigSpace.hyperparameters.uniform_integer.UniformIntegerHyperparameter:
                        type_thing = Integer
                        self._param_dict[f"{name}"] = {'type': type_thing, 'lower': param.lower, 'upper': param.upper,
                                                        'default': param.default_value, 'log': param.log}
                    case ConfigSpace.hyperparameters.categorical.CategoricalHyperparameter:
                        type_thing = Categorical
                        self._param_dict[f"{name}"] = {'type': type_thing, "set": list(param.choices), "default":param.default_value}
                        # breakpoint()
                        # {'type': Categorical, 'set': [0, 1], 'default': 1}
                        # {'type': Categorical, 'set': ["-luby", "-no-luby"], 'default': "-no-luby"}
                    case _:
                        print("No match found???")
                        sys.exit()


                # type_thing = Float
                # self._param_dict[f"-{name}"] = {'type': type_thing, 'lower': param.lower, 'upper': param.upper, 'default': param.default_value, 'log': param.log}
            # breakpoint()
            # self._configspace = ConfigurationSpace(seed=seed)

        else:
            print("ERROR: unrecognized solver: {}".format(self._name))
            exit()    


    @property
    def name(self):
        return self._name
    
    @property
    def configspace(self):
        return self._configspace

    
    def config_string(self, config):
        """ need to specify the command line argument format for each solver here """
        
        arguments = []        
        if self._name == 'Spear-32_1.2.1':
            for k, v in config.items():
                arguments.append(k + "=" + str(v))
        elif self._name == 'cryptominisat':
            for k, v in config.items():
                arguments.append(k + "=" + str(v))
            arguments.append("--printsol=0") # suppress printing the solution
        elif self._name == 'lingeling':
            for k, v in config.items():
                arguments.append("--" + k + "=" + str(v))
        elif self._name == 'glucose_static':
            for k, v in config.items():
                if hasattr(self.configspace[k], 'choices'): # categorical hyperparameter 
                    arguments.append(str(v))
                else:
                    arguments.append(k + "=" + str(v))
        return " ".join(arguments)


    def verify_instance(self, instance_loc: str, cutoff, config: ConfigSpace, output_queue, seed): # TODO: seed

        wrapped_runner = PipelineWrapper()

        instance_path = instance_loc
        specifics = '0'
        cutoff = str(cutoff)
        runlength = '2147483647'
        seed = '-1'
        start = ['--runsolver-path',
                 # os.getenv("RUNSOLVER_LOC"),
                 os.getenv("RUNSOLVER_LOC"),
                 instance_path, specifics, cutoff, runlength, seed,
                 '-threads', '1' # TODO: is this correct, it wasn't there before but im not sure if its necesasry
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
            wrapped_runner._ta_runtime = int(int(os.getenv("ORACLE_CAPTIME"))) # TODO: this doesnt really matter with utility func as util will be 0, but does matter in smac....


        sys.argv = tmp #ys.argv[:1]

        if wrapped_runner._ta_status in ["CRACHED"]:
            wrapped_runner._ta_runtime = int(os.getenv("ORACLE_CAPTIME")) #10 # cutoff time

        # output_queue.put((wrapped_runner._ta_status, float(wrapped_runner._ta_runtime),
        #                   str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality),
        #                   str(wrapped_runner._seed)))
        return (wrapped_runner._ta_status, float(wrapped_runner._ta_runtime),
                          str(wrapped_runner._ta_runlength), str(wrapped_runner._ta_quality),
                          str(wrapped_runner._seed))
    


    def dorun(self, config, instance, seed, captime, timemode="cpu"):

        """ """

        if self._name == "MipVerify":
            # ...
            ta_status, runtime, ta_runlength, ta_quality, seed = self.verify_instance(instance, captime, config, output_queue=None, seed=seed)
            # breakpoint()
            return runtime
            # breakpoint()
        else:
            command = self._callstring.format(self.config_string(config), instance, seed, captime, timemode).split(" ")
            proc = subprocess.run(command, capture_output=True) # run the command

            if proc.stdout.find(b"Maximum CPU time exceeded") >= 0: # runsolver will print this if it times out
                return captime
            elif proc.stdout.find(b"SATISFIABLE") < 0 and proc.stdout.find(b"UNSATISFIABLE") < 0: # something went wrong with the solver
                print(f"SOMETHING WENT WRONG executing run")
            else:
                try:
                    s1 = proc.stdout.split(b"Real time (s): ")[-1]
                    s2 = s1.split(b"CPU time (s): ")[1]
                    # wall_time = float(s1.split(b"\n")[0])
                    cpu_time = float(s2.split(b"\n")[0])
                    return min(cpu_time, captime - 1e-8) # make sure returned time is less than captime
                except IndexError:
                    print(f"SOMETHING WENT WRONG parsing")


    def construct_parameters(self):
        """ Build list of parameter objects from dict """
        params = []
        for name, param in self._param_dict.items():
            if param['type'] == Categorical:
                parameter = param['type'](name, param['set'], default=param['default'])
            else:
                parameter = param['type'](name, (param['lower'], param['upper']), default=param['default'], log=param['log'])
            params.append(parameter)
        return params


    def sample_best_neighbour(self, config, model):

        """ from SMAC paper: "including the set of all configurations that differ in the value of exactly one discrete parameter, 
        as well as four random neighbours for each numerical parameter" """

        best_val_so_far = 0
        best_neighbour_so_far = None

        for hp in config:

            param_type = self._param_dict[hp]['type']

            if param_type is Categorical:
                param_values = copy(self._param_dict[hp]['set'])
                param_values.remove(config[hp])
            
            elif param_type is Integer:
                upper = self._param_dict[hp]['upper']
                lower = self._param_dict[hp]['lower']
                is_log = self._param_dict[hp]['log']

                if upper - lower > 200: # If there are more than 200 possible values, sample as for floats 
                    
                    if is_log:
                        up = math.log(upper)
                        lo = math.log(lower)
                        mu = math.log(config[hp])
                    else:
                        up = upper
                        lo = lower
                        mu = config[hp]
                    
                    num_neighbours = 4 # as per SMAC
                    stdev = 0.2 * (up - lo)  # as per SMAC
                    a = (lo - mu) / stdev
                    b = (up - mu) / stdev

                    param_values = truncnorm.rvs(a, b, loc=mu, scale=stdev, size=num_neighbours)
                    if is_log:
                        param_values = [round(math.exp(v)) for v in param_values] # round back to integer 
                    else:
                        param_values = [round(v) for v in param_values] # round back to integer 
                    param_values = [min(v, upper) for v in param_values] # ensure within valid limits
                    param_values = [max(v, lower) for v in param_values] # ensure within valid limits

                else:
                    param_values = list(range(lower, upper + 1))
                    param_values.remove(config[hp])

            elif param_type is Float:
                upper = self._param_dict[hp]['upper']
                lower = self._param_dict[hp]['lower']
                is_log = self._param_dict[hp]['log']
                
                if is_log:
                    up = math.log(upper)
                    lo = math.log(lower)
                    mu = math.log(config[hp])
                else:
                    up = upper
                    lo = lower
                    mu = config[hp]

                num_neighbours = 4 # as per SMAC
                stdev = 0.2 * (up - lo)  # as per SMAC
                a = (lo - mu) / stdev
                b = (up - mu) / stdev

                param_values = truncnorm.rvs(a, b, loc=mu, scale=stdev, size=num_neighbours)
                if is_log:
                    param_values = [math.exp(v) for v in param_values]

            else:
                print(f"ERROR: Param type {param_type} not recognized")
                exit()

            for val in param_values: # take one neighbour step in each parameter dimension 
                new_config = deepcopy(config)
                new_config[hp] = val
                
                y_hat = model.predict(new_config)

                if y_hat > best_val_so_far: # keep the one that is predicted to be best 
                    best_neighbour_so_far = new_config
                    best_val_so_far = y_hat
            
        return best_neighbour_so_far






