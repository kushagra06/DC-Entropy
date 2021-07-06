# import itertools
import numpy as np 
# import gym
# import time
# import spinup.algos.pytorch.sac.core as core


def main():
    num_s = 12
    num_a = 4
    gamma = 0.99

    # b0 = np.array([0.0, 0.0, 0.0, 0.0, 
    #                0.0, 0.0, 0.0, 0.0, 
    #                1.0, 0.0, 0.0, 0.0])

    T = np.load("T.npy")

    np.random.seed(0)
    # pi = []
    # for s in range(num_s):
    #     x = np.random.rand(num_a)
    #     x = x/sum(x)
    #     pi.append(x)
    print(T)
    pi = np.ones([12, 4])/4
    # pi = np.random.choice(4, size=(12))
    print(pi)
    print(T[9][11][3])
    # Q = np.zeros((12, 4))


    # A = np.zeros((12,12))

    # for s1 in range(num_s):
    #     A[s1][0] = 1.0
    #     for s2 in range(1, num_s):
    #         if s1 != s2:
    #             temp_sum = sum(T[s1][s2] * pi[s2])
    #             A[s1][s2] = - gamma * temp_sum

    # dsa = pi
    # ds = np.random.rand(num_s)
    # ds = ds/sum(ds)
    
    # Q = np.zeros((num_s, num_a))
    
    # x = np.linalg.solve(A, b0)
    # print(x)

if __name__ == "__main__":
    main()