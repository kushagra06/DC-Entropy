import numpy as np

def occ_meas_nogym():
	num_s, num_a, gamma = 12, 4,  0.99
	T = np.load("T.npy")
	pi = np.ones([num_s, num_a]) / num_a
	b0 = np.zeros(num_s)
	b0[0] = 1.
	P = np.zeros([num_s, num_s])
	for s in range(num_s):
		for s_ in range(num_s):
			P[s][s_] = np.sum(T[s][s_] * pi[s])
	print(P)
	I = np.eye(num_s)
	A = I - gamma * P
	x = np.linalg.solve(A, b0)
	print(x)
	print(sum(x))
	return x

def occ_meas_gym():
	import gym
	# from gridworld import GridworldEnv
	# env = GridworldEnv()

	import gym_gridworlds
	env_name = 'Gridworld-v0'
	env = gym.make(env_name)
	num_s, num_a, gamma = 15, 4, 0.99
	# print(env.P)
	pi = np.ones([num_s, num_a]) / num_a
	# b0 = np.ones(num_s)
	b0 = np.zeros(num_s)
	b0[0] = 1.
	t_mat = np.zeros([num_s, num_s])
	for s in range(num_s):
		for s_ in range(num_s):
			t_mat[s][s_] = np.sum(env.P[:,s,s_] * pi[s])
	# print(t_mat)
	I = np.eye(num_s)
	A = I - gamma * t_mat
	print(A)
	# x = np.linalg.solve(A, b0)
	# print(x)
	# print(sum(x))
	# return x

def main():
	# x1 = occ_meas_nogym()
	x2 = occ_meas_gym()

if __name__ == '__main__':
	main()