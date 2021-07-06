# from __future__ import print_function
import numpy as np
import gym
from scipy.special import entr

import gym_gridworlds
# from gridworld import GridworldEnv

# import pyOpt
from scipy import optimize
import matplotlib.pyplot as plt 

env_name = 'Gridworld-v0'
env1 = gym.make(env_name)
# total_actions = GridworldEnv()
total_states, total_actions = env1.observation_space.n, env1.action_space.n
np.random.seed(0)




def obj_fun(pi, **kwargs):
	term1 = 0.0
	term2 = 0.0
	pi_k = kwargs['pi_k']
	q_k = kwargs['q_k']
	ds_k = kwargs['ds_k']
	dsa_k = kwargs['dsa_k']

	i, j = 0, 0
	for s in range(total_states):
		# H_k = entr(pi_k[s]).sum()
		# H = entr(pi[j:j+4]).sum()
		# logH = 0 if (H==0 or np.isinf(H)) else np.log(H)
		term2 +=  0#logH * ds_k[s] * H_k 
		j += 4
		for a in range(total_actions):
			logpi = 0 if pi[i]<=0 else np.log(pi[i])
			term1 += logpi * dsa_k[s][a] * (q_k[s][a])# + H_k)
			i += 1
	f = -(term1 + term2)
	g = [0] * total_states * 2
	i = 0
	for s in range(total_states):
		g[s] = sum(pi[i:i+4]) - 1
		g[s+1] = - g[s]
		i += 4
	fail = 0

	return f, g, fail

def pol_solve_pyOpt(pi_k, q_k, gamma):
	total_vars = pi_k.size
	total_constraints = total_states
	vars_per_constraint = total_actions
	ds_k, dsa_k = get_occupancy_measure(pi_k, gamma)
	lower_bound, upper_bound = [0]*total_vars, [1]*total_vars
	# eq = [1]*total_constraints

	opt_prob = pyOpt.Optimization('DC Entropy', obj_fun)
	opt_prob.addObj('f')
	opt_prob.addVarGroup('pi', total_vars, 'c', lower=lower_bound, upper=upper_bound)
	opt_prob.addConGroup('g', total_constraints*2)
	print(opt_prob)

	conmin = pyOpt.CONMIN()
	# conmin.setOption('IPRINT', 0)
	conmin(opt_prob, sens_type='FD', pi_k=pi_k, q_k=q_k, ds_k=ds_k, dsa_k=dsa_k)
	print(opt_prob.solution(0))
	# print(sol)

	return np.zeros(pi_k.size), 0

def get_constraints_matrix(total_constraints, total_vars, vars_per_constraint):
	constraints_mat = []
	for i in range(0, total_vars, vars_per_constraint):
		row = list(np.zeros(total_vars))
		for j in range(i, i+4):
			row[j] = 1
		constraints_mat.append(row)

	return constraints_mat

def pi_constraints(pi):
	total_constraints = total_states
	vars_per_constraint = total_actions
	total_vars = total_states * total_actions
	b = np.ones(total_constraints)
	constraints_mat = get_constraints_matrix(total_constraints, total_vars, vars_per_constraint)
	A = np.array(constraints_mat)
	ret = A.dot(pi) - b
	return ret

def get_occupancy_measure(pi_k, gamma=1.0):
	num_s = env1.observation_space.n
	num_a = env1.action_space.n
	b0 = np.ones(num_s) / num_s
	c = np.zeros(num_s)
	t_mat = np.zeros([num_s, num_s])
	for j in range(num_s):
		for s in range(num_s):
			val = 0
			for a in range(num_a):
				val += env1.P[a,s,j] * pi_k[s,a]
			c[s] = val
		t_mat[j] = c
	I = np.eye(num_s)
	A = I - gamma * t_mat
	x = np.linalg.solve(A, b0)

	x_sa = np.zeros([num_s, num_a])
	for s in range(num_s):
		for a in range(num_a):
			x_sa[s,a] = x[s] * pi_k[s,a]

	return x, x_sa

def pol_loss(pi, pi_k, q_k, ds_k, dsa_k):
	term1 = 0.0
	term2 = 0.0
	i, j = 0, 0
	for s in range(total_states):
		H_k = entr(pi_k[s]).sum()
		H = entr(pi[j:j+4]).sum()

		# logH = 0 if (H<=0 or np.isinf(H) or np.isnan(H)) else np.log(H)
		# logH_k = 0 if (H_k<=0 or np.isinf(H_k) or np.isnan(H_k)) else np.log(H_k)
		term2 += 0#(logH - logH_k) * ds_k[s] * H_k 
		j += 4
		for a in range(total_actions):
			# logpi = 0.0 if pi[i]<=0 else np.log(pi[i])
			# logpi_k = 0.0 if pi_k[s][a]<=0 else np.log(pi_k[s][a])
			logpi = 0 if pi[i]==0 else np.log(pi[i])
			logpi_k = 0 if pi_k[s][a]==0 else np.log(pi_k[s][a])
			term1 += (logpi - logpi_k) * dsa_k[s][a] * (q_k[s][a])#  + H_k)
			i += 1
	
	return -(term1 + term2)

def pol_solve(pi_k, q_k, gamma=1.0): 	
	total_vars = total_states * total_actions
	total_constraints = total_states
	vars_per_constraint = total_actions 

	ds_k, dsa_k = get_occupancy_measure(pi_k, gamma)
	
	# pi0 = pi_k	
	pi0 = pi_k.flatten()
	constraints_mat = get_constraints_matrix(total_constraints, total_vars, vars_per_constraint)
	b = np.ones(total_constraints) #pi sum up to 1
	# cons = {'type':'eq', 'fun':pi_constraints, 'args':(constraints_mat, b)}
	cons = {'type':'eq', 'fun':pi_constraints}

	bnds = [(0+1e-8, 1+1e-8)] * total_vars
	
	# lower_bound, upper_bound = np.zeros(total_vars), np.ones(total_vars)
	# bounds = optimize.Bounds(lower_bound, upper_bound) # inidividual pi_i between 0 and 1
	# linear_constraint = optimize.LinearConstraint(constraints_mat, b, b)
	# hess = lambda a, b, x: np.zeros((total_vars, total_vars))
	# res = optimize.minimize(pol_loss, x0=pi0, args=(pi_k, q_k, ds_k, dsa_k), method='trust-constr', constraints=linear_constraint, bounds=bnds, options={'disp':True})

	res = optimize.minimize(pol_loss, x0=pi0, args=(pi_k, q_k, ds_k, dsa_k), method='SLSQP', constraints=cons, bounds=bnds, options={'disp':True})
	obj_val = pol_loss(res.x, pi_k, q_k, ds_k, dsa_k)

	# res = optimize.fmin_slsqp(func=pol_loss, args=(pi_k, q_k, ds_k, dsa_k), x0=pi0, f_eqcons=pi_constraints, bounds=bnds)
	return res.x, obj_val
	# return res.out, res.fx

def pol_eval(pi, gamma=1.0):
	v = np.zeros(total_states)
	q = np.zeros((total_states, total_actions))
	eps = 1e-8
	
	while True:
		delta = 0
		for s in range(total_states):
			v_val = 0
			for a in range(total_actions):
				q_val = 0
				for (s_, p) in enumerate(env1.P[a][s]):
					r = 1 if s==0 else 0
					v_val += pi[s][a] * p * (r + gamma * v[s_])
					q_val += p * (r + gamma * sum(pi[s_] * q[s_]))
				q[s][a] = q_val
			delta = max(delta, np.abs(v_val - v[s]))
			v[s] = v_val

		if delta < eps:			
			break

	return q, v


######## can be formulated as a LP ###############
def soft_pol_eval(pi, gamma):
	v = np.zeros(total_states)
	q = np.zeros([total_states, total_actions]) #	- np.ones([total_states, total_actions])
	eps = 1e-8

	while True:
		delta = 0
		for s in range(total_states):
			v_val = 0
			logpi = np.zeros(total_actions)
			for a in range(total_actions):
				logpi[a] = 0 if pi[s,a]<=0 else np.log(pi[s,a])
				q_val = 0
				for (s_, p) in enumerate(env1.P[a][s]):
					r = 1 if s_==0 else 0
					q_val += r + gamma * p * v[s_]
				q[s][a] = q_val
			
			for a in range(total_actions):
				v_val += pi[s,a] * (q[s][a] - logpi[a])

			delta = max(delta, np.abs(v_val - v[s]))
			v[s] = v_val
		
		if delta < eps: 
			break

	return q, v

########### OLD SOFT POL EVALUATOIN ####################
# def soft_pol_eval(pi, gamma):

# 	v = np.zeros(total_states)

# 	q = np.zeros([total_states, total_actions]) #	- np.ones([total_states, total_actions])

# 	eps = 1e-5

# 	while True:

# 		delta = 0

# 		for s in range(total_states):

# 			v_val = 0

# 			for a in range(total_actions):

# 				q_val = 0

# 				for (s_, p) in enumerate(env1.P[a][s]):

# 					r = 1 if s==0 else 0

# 					q_val += r + gamma * p * v[s_]

# 					logpi = 0 if pi[s,a]<=0 else np.log(pi[s,a])

# 					v_val += pi[s,a] * (q[s][a] - logpi)

# 				q[s][a] = q_val

# 			delta = max(delta, np.abs(v_val - v[s]))

# 			v[s] = v_val

# 		if delta < eps: 

# 			break

# 		return q, v
#####################################################
def get_mdp_obj(pi, gamma=0.99):
	mdp_obj = 0
	ds, dsa = get_occupancy_measure(pi, gamma)
	for s in range(total_states):
		H = 0#entr(pi[s]).sum()
		r = 1 if s==0 else 0
		for a in range(total_actions):
			mdp_obj += dsa[s][a] * (r + H)

	return mdp_obj

def solve(gamma=1.0):
	total_epi = 100
	pi = np.ones([total_states, total_actions])/total_actions 
	obj_vals = []
	mdp_obj_vals = []
	b0 = np.ones(total_states) / total_states
	for epi in range(total_epi):
		q, v = pol_eval(pi, gamma)
		# q, v = soft_pol_eval(pi, gamma)
		new_pi, obj_val = pol_solve(pi, q, gamma)

		# new_pi, obj_val = pol_solve_pyOpt(pi, q, gamma)
		new_pi = np.reshape(new_pi, (total_states, total_actions))
		mdp_obj = get_mdp_obj(pi, gamma)

		if np.linalg.norm(pi-new_pi)  <= 1e-5:
			print('Converged at episode %d' %(epi+1))
			break
		print("Episode {}".format(epi))
		# print("pi: ",pi)
		# print("new_pi: ", new_pi)
		# print("v: ", v)
		print("mdp val: ", mdp_obj)
		print(np.sum(v * b0))
		obj_vals.append(obj_val)
		mdp_obj_vals.append(mdp_obj)
		print("\n")
		pi = np.copy(new_pi)
		if epi%50==0:
			print("EPISODE {}/{}".format(epi, total_epi))
		
		if abs(pol_obj_val - oldpol_obj_val) <= tol_obj:
			print('Converged at iteration {}'.format(itr))
			break
	return pi, q, v, obj_vals, mdp_obj_vals

def test_pol(pi):
	hits = 0
	total_steps = []
	total_epi = 100
	total_a = len(pi[0])
	start = 0
	for epi in range(total_epi):
		s = start%15
		steps = 0
		while True:
			a = np.random.choice(total_a, p=pi[s])
			next_s, r, done, _ = step(a)
			if next_s == 15:
				next_s = 0
			steps += 1
			if done:
				total_steps.append(steps)
				hits += 1
				start += 1
				break
			s = next_s
	print(total_steps)
	print("Avg steps taken: {}".format(np.nanmean(total_steps)))
	print("Hits: {} percent".format((hits/total_epi)*100))

def best_pol_directions(pi):
	d = {}
	d[0] = 'UP'
	d[1] = 'RIGHT'
	d[2] = 'DOWN'
	d[3] = 'LEFT'
	directions = []
	for s in range(total_states):
		a = np.argmax(pi[s])
		directions.append(d[a])

	return directions

def plotting_util(obj_vals, xlabel, ylabel, figdir, figname, dosave):
	fig, ax = plt.subplots()
	ax.plot(obj_vals)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if dosave:
		plt.savefig(figdir+figname)

def main():
	gamma = 0.90
	opt_pi, opt_q, opt_v, obj_vals, mdp_obj_vals = solve(gamma)
	
	print("OPT POLICY: ", opt_pi)
	print("OPT Q: ", opt_q)
	print("OPT V: ", opt_v)
	print("\n")
	b0 = np.ones(total_states) / total_states
	ds, dsa = get_occupancy_measure(opt_pi, gamma)
	print(np.sum(opt_v * b0))
	best_directions = best_pol_directions(opt_pi)
	for (s, d) in enumerate(best_directions):
		print("State {} : {}".format(s,d))
	
	plotting_util(obj_vals, xlabel="Number of iterations", ylabel="DC (policy) objective", figdir="/home/quark/dc_entropy/tabular/", figname="dc_obj_noH.png", dosave=True)
	plotting_util(mdp_obj_vals, xlabel="Number of iterations", ylabel="MDP objective", figdir="/home/quark/dc_entropy/tabular/", figname="mdp_obj_noH.png", dosave=True)

if __name__ == '__main__':
	main()
