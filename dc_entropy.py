from copy import deepcopy
import itertools
import numpy as np
import spinup.algos.pytorch.sac.core as core
import torch
from spinup.utils.logx import EpochLogger
import time
import gym
from torch.optim import Adam

# gym.logger.set_level(40)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) 
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # idxs = np.repeat(idxs, 8)
        batch = dict(obs=self.obs_buf[idxs], obs2=self.obs2_buf[idxs], act=self.act_buf[idxs], rew=self.rew_buf[idxs], done=self.done_buf[idxs])
        return {k:torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def dc_ent(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, logger_kwargs=dict(), 
    replay_size=int(1e6), steps_per_epoch=1000, epochs=20, gamma=0.99, polyak=0.995, lr=1e-2, 
    alpha=1.0, batch_size=32, start_steps=1000, update_after=500, update_every=50, num_test_episodes=10, max_ep_len=1000, save_freq=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape 
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters())#, ac.q2.parameters())

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1])#, ac.q2])
    logger.log('\n Number of parameters: \t pi: %d, \t q1: %d,\n' %var_counts)# \t q2: %d\n'%var_counts)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        #q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            #q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = q1_pi_targ #torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        # loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 #+ loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy())#,
                      # Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        #q2_pi = ac.q2(o, pi)
        q_pi = q1_pi#torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        # loss_pi = (alpha * logp_pi - q_pi).mean()
        with torch.no_grad():
            temp_loss_pi = (q_pi - alpha * logp_pi)
            # temp_mean = temp_loss_pi.mean()
            # temp_loss_pi = temp_loss_pi - temp_mean

        # temp_loss_pi = (q_pi - alpha * logp_pi)
        # temp_mean = temp_loss_pi.mean()
        # temp_loss_pi = (temp_loss_pi - temp_mean).detach()
        loss1_pi = (logp_pi * temp_loss_pi).mean()
        # loss2_pi = 0#alpha * (-logp_pi).mean()
        loss_pi = - loss1_pi #- loss2_pi
        
        # # Useful info for logging
        # pi_info = dict(LogPi=logp_pi.numpy())
        pi_info = dict(LogPi=q1_pi.detach().numpy())
        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr/10)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        # torch.nn.utils.clip_grad_value_(ac.pi.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), 3)
        # for p in ac.pi.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -3, 3))
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after or t % update_every == 0:
            for i in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env':env}, None)


            test_agent()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            #logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--exp_name', type=str, default='dc_ent')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    dc_ent(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)