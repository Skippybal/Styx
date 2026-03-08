import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver
from utils import parse_u, u_to_str, ensure_directory, random_seed, safe_save, colors, lw, fs

from smac.runhistory.runhistory import RunHistory


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('runids', help="strings identifying the COUP and SMAC runs", nargs="+", type=str)
    parser.add_argument('--imgdir', help="location of saved images", default="img/", type=str)

    args = parser.parse_args()

    ensure_directory(args.imgdir)

    print("Plotting utilities ... ")
    for runid in args.runids:

        validation_savepath = f"data/{runid}/validation_data.p"

        print("Loading run state ...")
        if "coup" in runid:
            state = pickle.load(open(f"data/{runid}/training_data.p", 'rb'))
            label = "COUP"
            c = colors[2]
        elif "smac" in runid:            
            state = pickle.load(open(f"data/{runid}/training_data.p", 'rb'))
            incumbents = [1 if i is None else i for i in state['i_stars']] # replace Nones with default config
            state['i_stars'] = incumbents
            label = "SMAC"
            c = colors[4]
        else:
            print(f"runid={runid} not recognized...")
            exit()

        validation_utilities = pickle.load(open(validation_savepath, 'rb'))
        plt.plot(state['observation_times_cpu'], [validation_utilities[i] for i in state['i_stars']], c=c, alpha=.5, linewidth=lw['main'], label=label)

    plt.ylim(0, 1)    
    plt.xlabel("CPU time (s)", fontsize=fs['axis'])
    plt.ylabel("incumbent utility", fontsize=fs['axis'])
    plt.legend()
    plt.savefig(args.imgdir + "_".join(args.runids) + "_utilities.pdf", bbox_inches='tight')
    plt.xscale('log')
    plt.xlim(1, plt.xlim()[1])
    plt.savefig(args.imgdir + "_".join(args.runids) + "_utilities_log.pdf", bbox_inches='tight')
    plt.clf()


    print("Plotting COUP epsilon, gamma ... ")

    runid_coup = [s for s in args.runids if "coup" in s][0]
    coup_state = pickle.load(open(f"data/{runid_coup}/training_data.p", 'rb'))

    epsilon_stars = coup_state['epsilon_stars']
    epsilon_primes = coup_state['epsilon_primes']
    gammas = coup_state['gammas']
    times = coup_state['observation_times_cpu'] 

    ## Smooth data:
    epsilon_stars_smooth = []
    eps0 = epsilon_stars[0]
    for i, eps in enumerate(epsilon_stars):
        if eps < eps0:
            eps0 = eps
        epsilon_stars_smooth.append(eps0)

    epsilon_primes_smooth = []
    eps0 = epsilon_primes[0]
    for i, eps in enumerate(epsilon_primes):
        if eps < eps0:
            eps0 = eps
        epsilon_primes_smooth.append(eps0)

    plt.plot(times, epsilon_stars_smooth, label="epsilon", c=colors[2], alpha=.5, linewidth=lw['main'])
    plt.plot(times, epsilon_primes_smooth, label="epsilon, random configs only", c=colors[2], alpha=.5, linewidth=lw['main'], linestyle='--')
    plt.plot(times, gammas, label="gamma", c=colors[3], alpha=.5, linewidth=lw['main'])
    plt.xlim(0, plt.xlim()[1])
    plt.ylim(0, .5)
    plt.xlabel("CPU time (s)", fontsize=fs['axis'])
    plt.ylabel("epsilon, gamma", fontsize=fs['axis'])
    plt.legend()
    plt.savefig(args.imgdir + "_".join(args.runids) + "_epsilongamma.pdf", bbox_inches='tight')
    plt.xscale('log')
    plt.xlim(1, plt.xlim()[1])
    plt.ylim(0, 1)
    plt.savefig(args.imgdir + "_".join(args.runids) + "_epsilongamma_log.pdf", bbox_inches='tight')
    plt.clf()


    print(f"Plots saved to {args.imgdir}")








