import os
import math
import time
import pickle
import numpy as np

from model import Model
from utils import random_seed, choose_max, safe_save


class COUP:
    """  """
    def __init__(self, solver, instances, u, delta=.01, n0=50, k0=1, state={}, savepath=None, usemodel=True, usehoeffding=False, useucb=False, maxseed=1e8):
        self.solver = solver
        self.instances = instances
        self.u = u
        self.delta = delta
        self.n0 = n0
        self.k0 = k0
        self.savepath = savepath        
        self.maxseed = maxseed
        self.usehoeffding = usehoeffding
        self.useucb = useucb
        if usemodel:
            self.model = Model(self.u, feature_names=[hp.name for hp in self.solver.configspace.values()])
            self.model_confidence = .1 # percentage for confidence bounds
        else:
            self.model = None
        
        self.init_state(state)

        if len(state) == 0: # starting a fresh run, not loading data from state
            for _ in range(self.n0):
                self.add_new_config()


    def init_state(self, state):
        """ """
        self.n = state.get('n', 0)
        self.k = state.get('k', {})
        self.r = state.get('r', 0)
        self.configs = state.get('configs', {})
        self.instances_run = state.get('instances_run', []) # instances that we ran (in order)
        self.random_seeds = state.get('random_seeds', [random_seed()]) # random seeds we have/will use 

        self.total_cpu_time = state.get('total_cpu_time', 0) # total CPU time
        self._wallclock_time = state.get('wallclock_time', 0) 

        self.i_stars = state.get('i_stars', [])
        self.epsilon_stars = state.get('epsilon_stars', [])
        self.epsilon_primes = state.get('epsilon_primes', [])
        self.gammas = state.get('gammas', [])
        self.observation_times_wallclock = state.get('observation_times_wallclock', []) # points in time where we observed i_star and epsilon_star
        self.observation_times_cpu = state.get('observation_times_cpu', []) # points in time where we observed i_star and epsilon_star
        self.is_from_random_sample = state.get('is_from_random_sample', []) # configs that sampled at random (not from model)

        self.U_hat = np.ones(self.n) * self.u(self.k0)
        self.F_hat = np.ones(self.n)        
        self.U_hat_ucb = np.ones(self.n)
        self.U_hat_lcb = np.ones(self.n) * self.u(self.k0)
        self.F_hat_lcb = np.zeros(self.n)
        self.UCB = np.ones(self.n)
        self.LCB = np.zeros(self.n)
        self.m = {}        

        for i in self.configs:
            self.m[i] = len(self.configs[i]['runtimes'])
            if self.m[i] > 0:
                self.U_hat[i] = np.mean([self.u(t) for t in self.configs[i]['runtimes'].values()])
                self.F_hat[i] = len(self.configs[i]['completed']) / len(self.configs[i]['runtimes'])
                self.update_bounds(i)

        self.t0 = time.time()


    def get_state(self):
        """ """
        state = {
            'n':self.n,
            'n0':self.n0,
            'k0':self.k0,
            'k':self.k,
            'r':self.r,
            'configs':self.configs,
            'random_seeds':self.random_seeds,
            'instances_run':self.instances_run,
            'total_cpu_time':self.total_cpu_time,
            'wallclock_time':self._wallclock_time,
            'i_stars': self.i_stars,
            'epsilon_stars': self.epsilon_stars,
            'epsilon_primes': self.epsilon_primes,
            'gammas': self.gammas,
            'observation_times_cpu': self.observation_times_cpu,
            'observation_times_wallclock': self.observation_times_wallclock,
            'is_from_random_sample': self.is_from_random_sample,
            't0':self.t0,
        }
        return state


    def save_state(self):
        """ """
        with open(self.savepath + '.tmp', 'wb') as f: # non-atomic write to temp file
            pickle.dump(self.get_state(), f)
        os.replace(self.savepath + '.tmp', self.savepath) # then atomic rename if successful


    def optimize(self, max_cpu_time=float('inf'), max_wallclock_time=float('inf'), savemod=50):
        """ COUP's main optimization loop """
        while True:
            try:
                if self.total_cpu_time >= max_cpu_time or self.wallclock_time >= max_wallclock_time:
                    print(f"\nCOUP reached max_cpu_time={max_cpu_time}, max_wallclock_time={max_wallclock_time} ...\n", flush=True)
                    self.update_output()
                    self.save_state()
                    print(self.message())
                    return

                if self.useucb: ## UCB algorithm 
                    i = choose_max(self.UCB, self.U_hat)
                    is_to_run = [i]

                else: ## LUCB algorithm:
                    # first we run argmax(U_hat)
                    i1 = np.argmax(self.U_hat)

                    # then we run argmax(UCB), excluding argmax(U_hat)
                    UCB_val = self.UCB[i1] # save value
                    self.UCB[i1] = 0 # set value to 0 to exclude i1=argmax(U_hat) from selection
                    i2 = choose_max(self.UCB, self.U_hat)
                    self.UCB[i1] = UCB_val # restore value 

                    is_to_run = [i1, i2]
                
                for i in is_to_run:
                    if self.doubling_condition(i): # if it's time to double k[i], rerun capped instances:
                        self.k[i] = 2 * self.k[i]
                        for instance_number in range(self.m[i]): # existing runs
                            if instance_number not in self.configs[i]['completed']: # rerun capped instance-seed pairs only
                                
                                self.run(i, instance_number)

                    # run the next instance:
                    if len(self.instances_run) <= self.m[i]:
                        self.instances_run.append(np.random.choice(len(self.instances))) # sample a random instance
                    if len(self.random_seeds) <= self.m[i]:
                        self.random_seeds.append(np.random.randint(low=0, high=self.maxseed)) # sample a random seed

                    self.run(i, self.m[i])

                    t = self.configs[i]['runtimes'][self.m[i]]
                    self.m[i] += 1
                    self.F_hat[i] = len(self.configs[i]['completed']) / self.m[i]
                    self.U_hat[i] = ((self.m[i] - 1) * self.U_hat[i] + self.u(t)) / self.m[i]
                    self.update_bounds(i)

                self.update_output()

                # save state
                if self.r % savemod == 0:
                    self.save_state()
                    print(self.message())

                # add new configurations
                # if self.epsilon_star**2 < self.gamma * (1 - np.max(self.UCB)): # condition from AAAI paper
                #     self.add_new_config()

                # add new configurations
                ucb_max = np.max(self.UCB[self.is_from_random_sample])
                lcb_max = np.max(self.LCB[self.is_from_random_sample])
                if (ucb_max - lcb_max)**2 < self.gamma_all_random * (1 - ucb_max): # condition for ony random configs
                    self.add_new_config()

                self.r += 1

            except KeyboardInterrupt:
                print(f"\n\nCOUP received KeyboardInterrupt. Saving state and returning.\n")
                self.update_output()
                self.save_state()
                print(self.message())
                return


    def update_bounds(self, i):
        """ """
        if self.usehoeffding:
            self.UCB[i] = min(self.U_hat[i] + (1 - self.u(self.k[i])) * self.alpha(i), self.UCB[i])
            self.LCB[i] = max(self.U_hat[i] - self.alpha(i) - self.u(self.k[i]) * (1 - self.F_hat[i]), self.LCB[i])
        else:
            # delta_i = self.delta / (35.61 * self.n**2 * self.m[i]**2 * (math.log2(self.k[i]) + 1)**2)
            delta_i = self.delta / (26.71 * self.n**2 * self.m[i]**2 * (math.log2(self.k[i]) + 1)**2) # don't need upper bound on F_hat
            a = math.log(1 / delta_i) / self.m[i]
            
            self.F_hat_lcb[i] = self.dkl_lcb(self.F_hat[i], a)
            
            U_hat_prime = max((self.U_hat[i] - self.u(self.k[i])) / (1 - self.u(self.k[i])), 0) # scale to [0,1]
            U_hat_ucb_prime = self.dkl_ucb(U_hat_prime, a)
            self.U_hat_ucb[i] = U_hat_ucb_prime * (1 - self.u(self.k[i])) + self.u(self.k[i]) # scale back to [u(k),1]
            
            U_hat_lcb_prime = self.dkl_lcb(U_hat_prime, a)
            self.U_hat_lcb[i] = U_hat_lcb_prime * (1 - self.u(self.k[i])) + self.u(self.k[i]) # scale back to [u(k),1]
            self.UCB[i] = min(self.U_hat_ucb[i], self.UCB[i])
            self.LCB[i] = max(self.U_hat_lcb[i] - self.u(self.k[i]) * (1 - self.F_hat_lcb[i]), self.LCB[i])


    def update_output(self):
        """ """
        self.i_stars.append(self.i_star)
        self.epsilon_stars.append(self.epsilon_star)
        self.epsilon_primes.append(self.epsilon_prime)
        self.gammas.append(self.gamma)
        self.observation_times_wallclock.append(self.wallclock_time)
        self.observation_times_cpu.append(self.total_cpu_time)


    def run(self, i, instance_number):
        """ """
        j = self.instances_run[instance_number]
        s = self.random_seeds[instance_number]

        t = self.solver.dorun(self.configs[i]['config'], self.instances[j], s, self.k[i])
        
        self.configs[i]['runtimes'][instance_number] = t
        self.total_cpu_time += t
        if t < self.k[i]:
            self.configs[i]['completed'][instance_number] = t



    def add_new_config(self, num_best_configs=10, num_random_configs=10000):
        """ """
        if len(self.configs) == 0: # if this is the first config added 
            print(f"COUP adding default config (i=0) ...")
            new_config = self.solver.configspace.get_default_configuration() # use the default confgiuration first
            self.is_from_random_sample.append(True)
        else:
            if self.model and self.n >= self.n0 and self.n % 2 == 0: # if there is a model and initial training data, use it to sample every other config
                print(f"COUP adding config number {self.n + 1} (i={self.n}) from model ...")
                self.model.train(self.configs)

                ## neighbourhood search as described in SMAC paper:
                best_configs = dict(sorted(self.configs.items(), key=lambda item: item[1]['model_selection_index'], reverse=True)[:num_best_configs])
                best_neighbours = []
                print("Sampling neighbours ...")
                for ci, c in enumerate(best_configs):
                    print(f"sampling neighbour {ci}")

                    best_neighbour = self.solver.sample_best_neighbour(self.configs[c]['config'], self.model)
                    y_hat_best = self.model.predict(best_neighbour, average=False) # get a list of predictions (one for each forest)
                    upper_bound_best = np.percentile(y_hat_best, (1 - self.model_confidence / 2) * 100)

                    while True:
                        contender = self.solver.sample_best_neighbour(best_neighbour, self.model)
                        y_hat_contender = self.model.predict(contender, average=False) # get a list of predictions (one for each forest)
                        upper_bound_contender = np.percentile(y_hat_contender, (1 - self.model_confidence / 2) * 100)
                        if upper_bound_contender > upper_bound_best: # found a better neighbour
                            print("found a better contender")
                            best_neighbour = contender
                            y_hat_best = y_hat_contender
                            upper_bound_best = upper_bound_contender
                        else: # finished searching for neighbours
                            break

                    best_neighbours.append(best_neighbour)

                print(f"Making predictions for {num_best_configs} best neighbours and {num_random_configs} random configs ...")

                candidates = best_neighbours + self.solver.configspace.sample_configuration(num_random_configs)                                              
                candidate_predictions = self.model.predict(candidates)
                best_prediction = np.argmax(candidate_predictions)
                new_config = candidates[best_prediction]
                self.is_from_random_sample.append(False)

            else:
                print(f"COUP adding config number {self.n + 1} (i={self.n}) from random sample ...")
                new_config = self.solver.configspace.sample_configuration() # sample a configuration 
                self.is_from_random_sample.append(True)

        self.configs[self.n] = {'config': new_config, 'runtimes': {}, 'completed': {}, 'model_selection_index': 0}
        self.m[self.n] = 0
        self.k[self.n] = self.k0

        self.U_hat = np.append(self.U_hat, self.u(self.k0))
        self.F_hat = np.append(self.F_hat, 0)
        self.UCB = np.append(self.UCB, 1)
        self.LCB = np.append(self.LCB, 0)
        self.U_hat_ucb = np.append(self.U_hat_ucb, 1)
        self.U_hat_lcb = np.append(self.U_hat_lcb, self.u(self.k0))
        self.F_hat_lcb = np.append(self.F_hat_lcb, 0)

        self.n += 1

        # print("")
        # print("m:")
        # print(dict(sorted(self.m.items(), key=lambda item: item[1], reverse=True)))
        # print("")

        # print("U_hat:")
        # uhat = dict([(i, self.U_hat[i]) for i in range(self.n)])
        # print(dict(sorted(uhat.items(), key=lambda item: item[1], reverse=True)))
        # print("")

        # print("UCB:")
        # ucb = dict([(i, self.UCB[i]) for i in range(self.n)])
        # print(dict(sorted(ucb.items(), key=lambda item: item[1], reverse=True)))
        # print("")

        # print("LCB:")
        # lcb = dict([(i, self.LCB[i]) for i in range(self.n)])
        # print(dict(sorted(lcb.items(), key=lambda item: item[1], reverse=True)))
        # print("")


    def d(self, p, q):
        """ KL-divergence term """
        if p == 0:
            return math.log2(1 / (1 - q))
        elif p == 1:
            return math.log2(1 / q)
        else:
            return p * math.log2(p / q) + (1 - p) * math.log2((1 - p) / (1 - q))


    def dkl_ucb(self, mu_hat, a, eps=1e-8, numpts=10000):
        """ Numerically solve for DKL-based upper bound """
        if mu_hat >= 1 - eps:
            return 1
        xs = np.linspace(mu_hat, 1 - eps, numpts)
        dkls = np.array([self.d(mu_hat, x) for x in xs])
        ucbs = xs[dkls < a]
        return ucbs[-1]


    def dkl_lcb(self, mu_hat, a, eps=1e-8, numpts=10000):
        """ Numerically solve for DKL-based lower bound """
        if mu_hat <= eps:       
            return 0
        xs = np.linspace(eps, mu_hat, numpts)
        dkls = np.array([self.d(mu_hat, x) for x in xs])
        lcbs = xs[dkls <= a]
        return lcbs[0]


    def alpha(self, i):
        """ Used for Hoeffding bounds """
        if self.m[i] == 0:
            return 1
        else:
            return math.sqrt(math.log(26.71 * self.n**2 * self.m[i]**2 * (math.log2(self.k[i]) + 1)**2 / self.delta) / 2 / self.m[i])


    def doubling_condition(self, i):
        """ """
        if self.usehoeffding:
            alpha = self.alpha(i)
            return (2 * (1 - self.u(self.k[i])) * alpha <= self.u(self.k[i]) * (1 - self.F_hat[i] + alpha))
        else:
            return self.U_hat_ucb[i] - self.U_hat_lcb[i] <= self.u(self.k[i]) * (1 - self.F_hat_lcb[i]) and self.m[i] > 0


    def instance(self, m):
        """ the instance from the m-th run """
        if m >= len(self.instances_run): # a new run
            self.instances_run.append(np.random.choice(len(self.instances))) # sample a random instance
        return self.instances_run[m]

    
    def random_seed(self, m): 
        """ the seed from the m-th run """
        if m >= len(self.random_seeds):
            new_seeds = random_seed(len(self.random_seeds)) # double number of seeds
            self.random_seeds = self.random_seeds + list(new_seeds)
        return self.random_seeds[m]
    

    def message(self):
        """ """
        out = "COUP: "
        out += f"r={self.r}. n={self.n}, gamma={self.gamma:.3f}, "
        out += f"i_star={self.i_star:5}, epsilon_star={self.epsilon_star:.3f}, epsilon_prime={self.epsilon_prime:.3f}, "
        out += f"add_cond=[{self.epsilon_star**2:.4f} < {self.gamma * (1 - np.max(self.UCB)):.4f}], "
        out += f"ucb=[{np.min(self.UCB):.3f}, {np.max(self.UCB):.3f}], lcb=[{np.min(self.LCB):.3f}, {np.max(self.LCB):.3f}], "
        out += f"U_hat=[{np.min(self.U_hat):.3f}, {np.max(self.U_hat):.3f}], F_hat=[{np.min(self.F_hat):.3f}, {np.max(self.F_hat):.3f}], "
        out += f"m=[{min(self.m.values())}, {max(self.m.values())}], k=[{min(self.k.values())}, {max(self.k.values())}], "        
        out += f"total_cpu_time={self.total_cpu_time:.1f}"
        return out


    def get_configuration(self, i):
        """ """
        return self.configs[i]['config']


    def guarantee(self):
        """ return the best guarantee COUP can make so far """
        xmin = np.argmin(np.array(self.epsilon_stars))
        return {'epsilon': self.epsilon_stars[xmin], 'gamma': self.gammas[xmin], 'i_star': self.i_stars[xmin]}


    @property
    def i_star(self):
        return choose_max(self.LCB, self.U_hat)


    @property
    def epsilon_star(self):
        return np.max(self.UCB) - np.max(self.LCB)


    @property
    def epsilon_prime(self, num_bootstrap_samples=10):
        if self.model is None:
            epsilons = np.ones(num_bootstrap_samples)
            # take bootstrap samples:
            for b in range(num_bootstrap_samples):
                ## include n0 original samples and a random half of other samples:
                random_half = ([True] * math.floor((self.n - self.n0) / 2)) + ([False] * math.ceil((self.n - self.n0) / 2))
                np.random.shuffle(random_half)                
                random_sample = [True for _ in range(self.n0)] + list(random_half)
                epsilons[b] = np.max(self.UCB[random_sample]) - np.max(self.LCB)
            return np.mean(epsilons)
        else:
            return np.max(self.UCB[self.is_from_random_sample]) - np.max(self.LCB)


    @property
    def gamma(self):
        if self.model is None:
            return np.log(math.pi**2 * self.n**2 / 3 / self.delta) / self.n
        else:
            n_random = math.floor((self.n - self.n0) / 2) + self.n0 # number of random sampled configs
            return np.log(math.pi**2 * n_random**2 / 3 / self.delta) / n_random


    @property
    def gamma_all_random(self):
        return np.log(math.pi**2 * self.n**2 / 3 / self.delta) / self.n
    

    @property
    def wallclock_time(self):
        t1 = time.time()
        self._wallclock_time += (t1 - self.t0)
        self.t0 = t1
        return self._wallclock_time




