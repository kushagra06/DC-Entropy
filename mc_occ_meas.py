import numpy as np
import gym

def get_transition_matrix():
	print(env.P)
	
def main():
	# num_s, num_a, gamma = 12, 4,  0.99
	# T = np.load("T.npy")
	T = get_transition_matrix()

	

if __name__ == '__main__':
	env_name = 'FrozenLake-v0'
	env = gym.make(env_name)
	main()