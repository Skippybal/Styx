# import sys
# import os
# import argparse
import math
# import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import colors as mplcolors
# from scipy.interpolate import interp1d

# import time

from utils import ecdf, u_satcomp, u_lin, u_step, u_exp, u_to_str, fs, lw, colors, ensure_directory




def solver_l1_distance(solvers, solver_base_rank):
	distance = 0
	for sid, solver in enumerate(solvers):
		rank_u = sid + 1
		rank_base = solver_base_rank[solver]
		distance += abs(rank_u - rank_base)
	return distance

	

def solver_color_by_rank(solvers):
	color_grid = [i / (len(solvers) - 1) for i in range(len(solvers))]
	solver_color = {}
	for i, solver in enumerate(solvers): 
		solver_color[solver] = color_grid[i]
	return solver_color


def u_parc(t, k, c):
	if t < k:
		return 1 - t / (c * k)
	else:
		return 0


def u_exp(t, lam):
	return np.exp(- t * lam)


if __name__ == "__main__":

	ensure_directory("img/satcomp/cdfs")
	ensure_directory("img/satcomp/utilities")

	data_path = "data/sc2023-detailed-results/results_main_detailed.csv"

	df_raw = pd.read_csv(data_path)
	df_raw = df_raw.drop(["hash", "vresult"], axis=1)

	par2means = df_raw.mean(axis=0).sort_values()
	solvers = list(par2means.index)

	df_raw[df_raw == 10000] = 5000 # "uncap" runs: runs that timed out at 5000s were counted as 10000

	print("SATCOMP solver ranking:")
	for i, s in enumerate(solvers):
		print(i+1, s)

	cmap = "viridis_r"
	solver_color = solver_color_by_rank(solvers)
	colormap = mpl.colormaps[cmap]


	####################### PARc u for different c #######################
	cs = [32, 4, 2, 1.5, 1]
	k = 5000
	tmax = k + 500
	color = colors[5]
	for i, c in enumerate(cs):
		linewidth = 2 * (len(cs) - i)
		msize = 20 * linewidth
		plt.plot([0, k], [1, 1 - 1 / c], linewidth=linewidth, color=color, zorder=1, label=f"c={c}")
		plt.scatter([k], [1 - 1 / c], s=msize, facecolors='white', edgecolors=color, zorder=2)
		plt.scatter([0, k], [1, 0], s=msize, color=color, zorder=2)		
		plt.plot([k, k], [1 - 1 / c, 0], linewidth=linewidth, color=color, linestyle='--', zorder=1)
	plt.xlim(0, tmax)
	plt.ylim(-.01, 1.02)
	plt.xlabel("time t (seconds)", fontsize=fs['axis'])
	plt.ylabel("utility u(t)", fontsize=fs['axis'])
	plt.title(f"$\\kappa = 5000$", fontsize=fs['axis'])
	plt.legend()
	plt.savefig("img/satcomp/u_diffc.pdf", bbox_inches='tight')
	plt.clf()


	####################### PARc u for different k #######################
	c = 2
	ks = [5000, 4000, 2000, 1000, 500, 100]
	tmax = k + 500
	color = colors[5]
	for i, k in enumerate(ks):
		# linewidth = 10 - 2 * i
		linewidth = (2 * (len(ks) - i))
		msize = 20 * linewidth
		plt.plot([0, k], [1, 1 - 1 / c], linewidth=linewidth, color=color, zorder=1, label=f"$\\kappa$={k}")		
		plt.scatter([0, k], [1, 0], s=msize, color=color, zorder=2)
		plt.scatter([k], [1 - 1 / c], s=msize, facecolors='white', edgecolors=color, zorder=2)
		plt.plot([k, tmax], [0, 0], linewidth=linewidth, color=color, zorder=1)
		plt.plot([k, k], [1 - 1 / c, 0], linewidth=linewidth, color=color, linestyle='--', zorder=1)
	plt.xlim(0, tmax)
	plt.ylim(-.01, 1.01)
	plt.xlabel("time t (seconds)", fontsize=fs['axis'])
	plt.ylabel("utility u(t)", fontsize=fs['axis'])
	plt.title(f"$c = 2$", fontsize=fs['axis'])
	plt.legend()
	plt.savefig("img/satcomp/u_diffk.pdf", bbox_inches='tight')
	plt.clf()



	# ####################### Give each solver a color #######################	
	# color_data = np.transpose([[solver_color[solver] for solver in solvers]])
	# plt.figure(figsize=(3, 7))
	# plt.pcolormesh(color_data, cmap=cmap)
	# plt.xticks([])
	# plt.yticks([])
	# for i, solver in enumerate(reversed(solvers)):
	# 	plt.text(1.01, i + 0.5, solver, horizontalalignment='left', verticalalignment='center', fontsize=10)
	# plt.savefig("img/solvercolor.pdf", bbox_inches='tight')
	# plt.figure()
	# plt.clf()
	

	####################### PARc rank for different c #######################
	cs = [1, 1.5, 2, 4, 32, 1e15]
	utility_functions = [(u_satcomp, {'T': 5000, 'c': c}) for c in cs]
	utility_functions.append((u_step, {'k0': 5000}))
	
	color_data = np.zeros((len(solvers), 1))
	solvers_sorted_by_u = []
	for ui, u in enumerate(utility_functions):
		u_fn, u_params = u
		u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
		df_u = df_raw.apply(u_vect, axis=0)	
		mean_utilities = df_u.mean(axis=0).sort_values(ascending=False)
		solvers_sorted = list(mean_utilities.index)
		color_data = np.concatenate((color_data, np.transpose([[solver_color[solver] for solver in reversed(solvers_sorted)]])), axis=1)
		solvers_sorted_by_u.append(solvers_sorted)
	color_data = color_data[:, 1:]

	plt.pcolormesh(color_data, cmap=cmap, vmin=0, vmax=1)
	cs[2] = "2(SAT Comp.)"
	cs[-1] = "1e15"
	plt.xticks([i + 0.5 for i in range(len(utility_functions))], cs + ["$\\infty$(step)"])
	plt.yticks([len(solvers) - .5, len(solvers) - 1.5, len(solvers) - 2.5], ["1st", "2nd", "3rd"], fontsize=7)
	plt.xlabel("c", fontsize=fs['axis'])
	plt.ylabel("ranking", fontsize=fs['axis'])
	plt.title("$\\kappa=5000$", fontsize=fs['axis'])
	plt.savefig("img/satcomp/rank_diffc.pdf", bbox_inches='tight')
	plt.clf()

	distances = np.zeros((len(utility_functions), len(utility_functions)))
	for ui, _ in enumerate(utility_functions):
		solver_base_rank = dict((solver, s + 1) for s, solver in enumerate(solvers_sorted_by_u[ui]))
		for uj, u in enumerate(utility_functions):
			distance = solver_l1_distance(solvers_sorted_by_u[uj], solver_base_rank)
			distances[ui, uj] = distance
	print("L1 distances, different c:")
	print(distances)



	####################### PARc rank for different k #######################
	ks = [5000, 4000, 2000, 1000, 500, 100, 10]
	utility_functions = [(u_satcomp, {'T': k, 'c': 2}) for k in ks]

	color_data = np.zeros((len(solvers), 1))
	solvers_sorted_by_u = []
	for ui, u in enumerate(utility_functions):
		u_fn, u_params = u
		u_vect = np.vectorize(lambda t: u_fn(t, **u_params), otypes=[float])
		df_u = df_raw.apply(u_vect, axis=0)	
		mean_utilities = df_u.mean(axis=0).sort_values(ascending=False)
		solvers_sorted = list(mean_utilities.index)
		color_data = np.concatenate((color_data, np.transpose([[solver_color[solver] for solver in reversed(solvers_sorted)]])), axis=1)
		solvers_sorted_by_u.append(solvers_sorted)
	color_data = color_data[:, 1:]

	plt.pcolormesh(color_data, cmap=cmap, vmin=0, vmax=1)
	plt.xticks([i + 0.5 for i in range(len(utility_functions))], ks)
	plt.yticks([len(solvers) - .5, len(solvers) - 1.5, len(solvers) - 2.5], ["1st", "2nd", "3rd"], fontsize=7)
	plt.xlabel("$\\kappa$", fontsize=fs['axis'])
	plt.ylabel("ranking", fontsize=fs['axis'])
	plt.title("$c=2$", fontsize=fs['axis'])
	plt.savefig("img/satcomp/rank_diffk.pdf", bbox_inches='tight')
	plt.clf()

	distances = np.zeros((len(utility_functions), len(utility_functions)))
	for ui, _ in enumerate(utility_functions):
		solver_base_rank = dict((solver, s + 1) for s, solver in enumerate(solvers_sorted_by_u[ui]))
		for uj, u in enumerate(utility_functions):
			distance = solver_l1_distance(solvers_sorted_by_u[uj], solver_base_rank)
			distances[ui, uj] = distance
	print("L1 distances, different k:")
	print(distances)




	####################### What are runtime cdfs like? #######################

	xs0, ys0 = ecdf(df_raw[solvers[0]].values)
	crosspoints = {}
	for si, solver in enumerate(solvers[1:]):
		xs1, ys1 = ecdf(df_raw[solver].values)
		for i, y in enumerate(reversed(ys0)):
			if xs0[i] > xs1[i]:
				crosspoints[solver] = ([xs0[i]], [1 - y])

	print("Max crosspoint:", max(cp[0][0] for cp in crosspoints.values()))

	print("Solvers that are FOSD by the winner:")
	for solver in solvers:
		if solver not in crosspoints:
			print(solver)

	print("plotting CDFs...")
	fig, ax = plt.subplots()
	ax.set_facecolor('lightgray')
	plt.grid(which='major', axis='both', color='gray', linestyle='--', linewidth=0.5)
	for solver in reversed(solvers):
		plt.step(*ecdf(df_raw[solver].values), color=colormap(solver_color[solver]))
	plt.step(*ecdf(df_raw[solvers[0]].values), color=colormap(solver_color[solvers[0]]), linewidth=lw['main'])
	plt.xlim(-100, 5100)
	plt.ylim(0, .75)
	plt.xlabel("t", fontsize=fs['axis'])
	plt.ylabel("F(t)", fontsize=fs['axis'])
	plt.savefig("img/satcomp/satcomp_cdfs.pdf", bbox_inches='tight')
	plt.xscale('log')
	plt.xlim(2e-3, 1e4)
	plt.savefig("img/satcomp/cdfs/cdfs_log.pdf", bbox_inches='tight')

	for solver, point in crosspoints.items():		
		plt.scatter(*point, color='black', s=200, marker='x', zorder=3)
	plt.savefig("img/satcomp/cdfs/cdfs_marks_log.pdf", bbox_inches='tight')
	plt.xscale('linear')
	plt.xlim(-100, 5100)
	plt.savefig("img/satcomp/cdfs/cdfs_marks.pdf", bbox_inches='tight')
	plt.clf()


	solvers_plot = solvers[1:]
	for si, solver in enumerate(solvers_plot):
		fig, ax = plt.subplots()
		ax.set_facecolor('lightgray')
		plt.grid(which='major', axis='both', color='gray', linestyle='--', linewidth=0.5)
		plt.step(*ecdf(df_raw[solvers[0]].values), color=colormap(solver_color[solvers[0]]), linewidth=lw['main'], label=f"Winner ({solvers[0]})")
		plt.step(*ecdf(df_raw[solver].values), color=colormap(solver_color[solver]), linewidth=lw['main'], label=f"Rank {si+2} ({solver})")
		plt.legend(loc='lower right')		
		plt.xlim(-100, 5100)
		plt.ylim(0, .75)
		plt.xlabel("t", fontsize=fs['axis'])
		plt.ylabel("F(t)", fontsize=fs['axis'])
		plt.savefig(f"img/satcomp/cdfs/cdfs_few_{si+2}.pdf", bbox_inches='tight')
		plt.xscale('log')
		plt.legend(loc='upper left')
		plt.xlim(2e-3, 1e4)
		if solver in crosspoints:
			plt.scatter(*crosspoints[solver], color='black', s=200, marker='x', zorder=3)
		plt.savefig(f"img/satcomp/cdfs/cdfs_few_log_{si+2}.pdf", bbox_inches='tight')
		plt.xscale('linear')
		plt.xlim(-100, 5100)
		plt.legend(loc='lower right')
		plt.savefig(f"img/satcomp/cdfs/cdfs_few_{si+2}_mark.pdf", bbox_inches='tight')

		plt.close()



	####################### utility regret as a function of params #######################
	
	scenarios = []
	scenarios.append(("diffk", "$\\kappa$", np.logspace(0, 3.7, 5000), lambda t, param: u_parc(t, param, 2)))
	scenarios.append(("diffc", "c", np.logspace(0, 2, 500), lambda t, param: u_parc(t, 5000, param)))
	scenarios.append(("exp", "$\\lambda$", np.logspace(-6, 4, 5000), lambda t, param: u_exp(t, param)))

	next_color = 0
	plotcolors = {}
	for scenario in scenarios:
		tag, param_name, params, u = scenario

		print(f"starting {tag}")

		mus_by_param = {}
		print("computing mus by param...")
		for param in params:
			mus_by_param[param] = {}
			u_vect = np.vectorize(lambda t: u(t, param), otypes=[float])
			for i, s in enumerate(solvers):
				runtimes = np.array(df_raw[s])
				mu = np.mean(u_vect(runtimes))
				mus_by_param[param][s] = mu

		optimals = []
		optimums = {}		
		print("computing optimals...")
		for param in params:
			opt = max(mus_by_param[param], key=mus_by_param[param].get)
			optimals.append(opt)
			optimums[param] = mus_by_param[param][opt]
			if opt not in plotcolors:
				plotcolors[opt] = colors[next_color]
				next_color += 1
		
		plt.scatter(params, optimals, c=[plotcolors[opt] for opt in optimals])
		plt.xlabel(param_name, fontsize=fs['axis'])
		plt.xticks(fontsize=fs['ticks'])
		plt.yticks(fontsize=fs['legend'])
		plt.xscale('log')
		plt.savefig(f"img/satcomp/utilities/{tag}_optimals.pdf", bbox_inches='tight')
		plt.clf()

		for i, s in enumerate(solvers):
			if s in optimals:
				runtimes = np.array(df_raw[s])
				mus = []
				for param in params:
					mus.append(optimums[param] - mus_by_param[param][s])
				# plt.plot(params, mus, label=f"Rank {i+1} ({s})", color=colormap(solver_color[s]), linewidth=lw["main"], zorder=(i+1))
				plt.plot(params, mus, label=f"Rank {i+1} ({s})", color=plotcolors[s], linewidth=lw["main"], zorder=(len(solvers) - i), alpha=.7)
		
		first_other_solver = 1
		for i, s in enumerate(solvers):
			if s not in optimals:
				runtimes = np.array(df_raw[s])
				mus = []
				for param in params:
					mus.append(optimums[param] - mus_by_param[param][s])
				plt.plot(params, mus, label="Other solvers" if first_other_solver == 1 else None, color='silver', linewidth=lw['small'], zorder=0)
				first_other_solver = 0 
		
		plt.xticks(fontsize=fs['legend'])
		plt.yticks(fontsize=fs['legend'])
		plt.xscale('log')
		plt.legend()
		plt.xlabel(param_name, fontsize=fs['axis'])
		plt.ylabel("Regret", fontsize=fs['axis'])
		plt.savefig(f"img/satcomp/utilities/{tag}_optimals_utility.pdf", bbox_inches='tight')
		plt.clf()











