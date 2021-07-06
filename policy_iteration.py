import gym
import numpy as np 
import gym_gridworlds
from gridworld import GridworldEnv

def run_epi(pi, gamma, render=False):
	s = 0
	total_r = 0.0
	t = 0.0
	while True:
		if render:
			env.render()
		next_s, r, done, _ = env.step(np.argmax(pi[s]))
		total_r += gamma**t + r 
		t += 1
		s = next_s
		if done:
			break
	return total_r

def eval_pol(pol, gamma, n=100):
	rewards = [run_epi(pol, gamma, render=False) for _ in range(n)]
	return np.mean(rewards)

# def pol_impr(q, gamma):
# 	for s in range(total_states):
# 		pol[s] = np.argmax(q[s])
# 	return pol

def pol_eval(pi, gamma=1.0):
	v = np.zeros(total_states)
	q = np.zeros((total_states, total_actions))
	eps = 1e-3

	while True:
		delta = 0
		for s in range(total_states):
			v_val = 0
			for a in range(total_actions):
				q_val = 0
				for p, s_, r, done in env.P[s][a]:
					v_val += pi[s][a] * p * (r + gamma * v[s_])
					q_val += p * (r + gamma * np.sum(pi[s_] * q[s_]))
				# for s_, p in enumerate(env.P[a][s]):
					# q_val += p * (env.R[a][s] + gamma * np.sum(pi[s_] * q[s_]))
					# v_val += pi[s][a] * p * (env.R[a][s] + gamma * v[s_])
					# q_val += p * (r + gamma * v_val)
				q[s][a] = q_val	
			# v[s] = np.sum(pi[s] * q[s])
			delta = max(delta, np.abs(v_val - v[s]))
			v[s] = v_val

		if delta < eps:			
			break

	return q, np.array(v)

def pol_iter(gamma=1.0):
	# pi = np.random.choice(total_actions, size=(total_states))
	pi = np.ones([total_states, total_actions])/total_actions
	total_iterations = 10
	for i in range(total_iterations):
		if i%100==0:
			print("EPISODE {}/{}".format(i, total_iterations))

		q, v = pol_eval(pi, gamma)
		
		print("pi before: ", pi)


		######## policy improvement ########
		pol_stable = True
		for s in range(total_states):
			best_a = np.argmax(q[s])
			chosen_a = np.argmax(pi[s])
			if best_a != chosen_a:
				pol_stable = False
			pi[s] = np.eye(total_actions)[best_a]
		
		print("pi after: ", pi)
		print("\n")
		if pol_stable:
			print("Policy iteration converged at step {}".format(i+1))
			break
		######## policy improvement ########


		# new_pi = pol_impr(q, gamma)
		# if (np.all(pi == new_pi)):
		# 	print("Policy iteration converged at step {}".format(epi+1))
		# 	break
		# pi = new_pi

	return pi, q, v 

def test_pol(pi, gamma):
	holes = 0
	total_steps = []
	s = env.reset()
	total_epi = 100
	for epi in range(total_epi):
		s = env.reset()
		steps = 0
		while True:
			a = np.random.choice(total_actions, p=pi[s])
			next_s, r, done, _ = env.step(a)
			steps += 1
			if done and r==1:
				total_steps.append(steps)
				break
			elif done and r==0:
				holes += 1
				break
			s = next_s
	print("Avg steps: ", np.nanmean(total_steps))
	print("Misses %: ", (holes/total_epi)*100)



def main():
	gamma = 0.99
	opt_pol, opt_q, opt_v = pol_iter(gamma)
	# rewards = eval_pol(opt_pol, gamma, n=100)
	print("OPT POL: ", opt_pol)
	print("OPT Q: ", opt_q)
	print("OPT V: ", opt_v)
	# print("Avg reward = {}".format(rewards))
	# test_pol(opt_pol, gamma)

if __name__ == '__main__':
	env_name = 'Gridworld-v0'
	# env = gym.make(env_name)
	env = GridworldEnv()
	total_states = env.observation_space.n
	total_actions = env.action_space.n
	main()