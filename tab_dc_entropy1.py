# from __future__ import print_function
import numpy as np
import gym
# from scipy.stats import entropy
from scipy.special import entr

# import pyOpt
# from pyOpt import SLSQP
from scipy import optimize

env = gym.make('FrozenLake-v0')
np.random.seed(0)


def get_constraints_matrix(total_constraints, total_vars, vars_per_constraint):
	constraints_mat = []
	for i in range(0, total_vars, vars_per_constraint):
		row = list(np.zeros(total_vars))
		for j in range(i, i+4):
			row[j] = 1
		constraints_mat.append(row)

	return constraints_mat

def pi_constraints(pi, constraints_mat, b):
	A = np.array(constraints_mat)
	ret = A.dot(pi) - b
	return ret

def get_occupancy_measure(pi_k, gamma=1.0):

	
	return ds_k, dsa_k

def pol_loss(pi, pi_k, q_k, ds_k, dsa_k):
	term1 = 0.0
	term2 = 0.0
	i, j = 0, 0
	for s in range(env.nS):
		H_k = 0#entr(pi_k[s]).sum()
		H = 0#entr(pi[j:j+4]).sum()
		# if H==0 or np.isnan(H) or np.isinf(H):
		# 	print("s = {}, H = {}".format(s, H))
		# 	print("pi: ", sum(pi[j:j+4]))
		# 	print("pi_k: ", sum(pi_k[s]))
		logH = 0 if (H==0 or np.isinf(H)) else np.log(H)
		term2 +=  0#logH * ds_k[s] * H_k 
		j += 4
		for a in range(env.nA):
			# logpi = 0.0 if (pi[i]<=0 or np.isinf(pi[i]) or np.isnan(pi[i])) else np.log(pi[i])
			logpi = 0.0 if pi[i]<=0 else np.log(pi[i])
			# if pi[i]<=0 or np.isinf(pi[i]) or np.isnan(pi[i]):
			# 	print("pi, s, a: ",pi[i], s, a)
			term1 += logpi * dsa_k[s][a] * (q_k[s][a] + H_k)
			i += 1
	return -(term1 + term2)

def pol_solve(pi_k, q_k, gamma=1.0): 	
	total_vars = pi_k.size
	total_constraints = env.nS
	vars_per_constraint = env.nA 

	ds_k, dsa_k = get_occupancy_measure(pi_k, gamma)
	
	# pi0 = np.zeros(total_vars)
	pi0 = np.ones([env.nS, env.nA])/env.nA #initial guess

	constraints_mat = get_constraints_matrix(total_constraints, total_vars, vars_per_constraint)
	b = np.ones(total_constraints) #pi sum up to 1
	cons = {'type':'eq', 'fun':pi_constraints, 'args':(constraints_mat, b)}
	bnds = [(0.0, 1.0)] * total_vars
	
	# lower_bound, upper_bound = np.zeros(total_vars), np.ones(total_vars)
	# bounds = optimize.Bounds(lower_bound, upper_bound) # inidividual pi_i between 0 and 1
	# linear_constraint = optimize.LinearConstraint(constraints_mat, b, b)
	# hess = lambda a, b, x: np.zeros((total_vars, total_vars))
	# res = optimize.minimize(pol_loss, x0=pi0, args=(pi_k, q_k, ds_k, dsa_k), method='SLSQP', constraints=linear_constraint, bounds=bounds)

	res = optimize.minimize(pol_loss, x0=pi0, args=(pi_k, q_k, ds_k, dsa_k), method='SLSQP', constraints=cons, bounds=bnds)

	return res.x


# def obj_fun(pi, **kwargs):
# 	term1 = 0.0
# 	term2 = 0.0
# 	pi_k = kwargs['pi_k']
# 	q_k = kwargs['q_k']
# 	ds_k = kwargs['ds_k']
# 	dsa_k = kwargs['dsa_k']

# 	i, j = 0, 0
# 	for s in range(env.nS):
# 		H_k = entr(pi_k[s]).sum()
# 		H = entr(pi[j:j+4]).sum()
# 		logH = 0 if (H==0 or np.isinf(H)) else np.log(H)
# 		term2 +=  logH * ds_k[s] * H_k 
# 		j += 4
# 		for a in range(env.nA):
# 			logpi = 0 if pi[i]<=0 else np.log(pi[i])
# 			term1 += logpi * dsa_k[s][a] * (q_k[s][a] + H_k)
# 			i += 1
# 	f = -(term1 + term2)
# 	g = [0] * env.nS
# 	i = 0
# 	for s in range(env.nS):
# 		g[s] = sum(pi[i:i+4]) - 1
# 		i += 4
# 	fail = 0

# 	return f, g, fail

# def pol_solve_pyOpt(pi_k, q_k, gamma):
# 	total_vars = pi_k.size
# 	total_constraints = env.nS
# 	vars_per_constraint = env.nA
# 	ds_k, dsa_k = get_occupancy_measure(pi_k, gamma)
# 	lower_bound, upper_bound = [0]*total_vars, [1]*total_vars
# 	eq = [1]*total_constraints

# 	opt_prob = pyOpt.Optimization('DC Entropy', obj_fun)
# 	opt_prob.addObj('f')
# 	opt_prob.addVarGroup('pi', total_vars, 'c', lower=lower_bound, upper=upper_bound)
# 	opt_prob.addConGroup('g', total_constraints, 'e')
# 	print(opt_prob)

# 	slsqp = pyOpt.SLSQP()
# 	slsqp.setOption('IPRINT', 0)
# 	sol = slsqp(opt_prob, pi_k=pi_k, q_k=q_k, ds_k=ds_k, dsa_k=dsa_k)

# 	print(sol)

# 	return np.zeros(pi_k.size)


def pol_eval(pi, gamma=1.0):
	v = np.zeros(env.nS)
	q = np.zeros((env.nS, env.nA))
	eps = 1e-5
	
	# while True:
	# 	# prev_v = np.copy(v)
	# 	delta = 0
	# 	for s in range(env.nS):
	# 		v_old = np.copy(v[s])
	# 		q[s] = [np.sum([p * (r + gamma * np.sum(pi[s_]*q[s_])) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
	# 		# q[s] = q_sa
	# 		v[s] = np.sum(q[s]*pi[s])
	# 		delta = max(delta, abs(v_old-v[s]))		
	# 	if delta < eps:
	# 		break
	
	while True:
		delta = 0
		for s in range(env.nS):
			v_old = np.copy(v[s])
			for a in range(env.nA):
				q_val = 0
				for p, next_s, r, done in env.P[s][a]:
					q_val += p * (r + gamma * sum(pi[next_s] * q[next_s]))
				q[s][a] = q_val
			v[s] = np.sum(pi[s] * q[s])
			delta = max(delta, np.abs(v[s] - v_old))

		if delta < eps:			
			break

	return q


def solve(gamma=1.0):
	total_epi = 5000
	# pi = np.random.choice(env.nA, size=(env.nS))
	pi = np.ones([env.nS, env.nA])/env.nA 
	
	# pi = np.zeros((env.nS, env.nA))
	
	for epi in range(total_epi):
		q = pol_eval(pi, gamma)
		new_pi = pol_solve(pi, q, gamma)
		# new_pi = pol_solve_pyOpt(pi, q, gamma)
		new_pi = np.reshape(new_pi, (env.nS, env.nA))
		if np.sum(np.fabs(new_pi - pi)) < 1e-4:
			print('Converged at episode %d' %(epi+1))
			break
		# print("Episode {}".format(epi))
		# print("pi: ",pi)
		# print("new_pi: ", new_pi)
		# print("\n")
		pi = new_pi
		if epi%100==0:
			print("EPISODE {}/{}".format(epi, total_epi))
	return pi, q

def test_pol(pi):
	holes = 0
	total_steps = []
	total_epi = 100
	total_a = len(pi[0])
	for epi in range(total_epi):
		s = env.reset()
		steps = 0
		while True:
			a = np.random.choice(total_a, p=pi[s])
			next_s, r, done, _ = env.step(a)
			steps += 1
			if done and r==1:
				total_steps.append(steps)
				break
			if done and r==0:
				holes += 1
				break
			s = next_s
	print("Avg steps taken: {}".format(np.nanmean(total_steps)))
	print("Misses: {} percent".format((holes/total_epi)*100))

def main():
	gamma = 0.99
	opt_pi, opt_q = solve(gamma)
	total_states = len(opt_pi)
	opt_v = np.zeros(total_states)
	for s in range(total_states):
		opt_v[s] = sum(opt_pi[s] * opt_q[s])
	print("OPT POLICY: ", opt_pi)
	print("OPT Q: ", opt_q)
	print("OPT V: ", opt_v)
	print("\n")

	test_pol(opt_pi)

if __name__ == '__main__':
	main()
