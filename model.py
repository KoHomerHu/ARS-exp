import numpy as np
import gymnasium as gym
import multiprocessing as mp
import copy

class Normalizer():
    def __init__(self, input_dim):
        self.n = np.zeros(input_dim)
        self.mean = np.zeros(input_dim)
        self.mean_diff = np.zeros(input_dim)
        self.var = np.zeros(input_dim)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(np.maximum(self.var, 1e-2))
        return (inputs - obs_mean) / obs_std
    

def roll_out(env_name, weight, normalizer, seed=42, horizon=2000, rendering=False):
    if rendering:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    np.random.seed(seed)
    env.action_space.seed(seed)
    s, _ = env.reset(seed=seed)
    total_reward = 0
    steps = 0

    s_lst = [s,]

    while steps < horizon:
        if rendering:
            env.render()
        norm_s = normalizer.normalize(s)
        a = weight.dot(norm_s)
        s, r, terminated, truncated, _ = env.step(a)
        s_lst.append(s) # save unnormalized state
        r = max(min(r, 1), -1) # reward clipping
        total_reward += r
        steps += 1
        if terminated or truncated:
            break

    return total_reward, s_lst
    

"""
Implementation of ARS V2.
Parameters: 
 - alpha: step-size
 - N: number of directions sampled per iteration
 - nu: standard deviation of the exploration nosie
 - b: number of top-performing directions to use
"""
class ARS():
    def __init__(self, input_dim, output_dim, alpha, N, nu, b, env_name, seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.N = N
        self.nu = nu
        self.b = b
        self.env_name = env_name
        self.seed = seed

        self.normalizer = Normalizer(input_dim)
        self.weight = np.zeros((self.output_dim, self.input_dim))

        self.num_cores = mp.cpu_count()

        np.random.seed(self.seed)

    def train_one_iter(self, num_iters):
        directions = np.random.standard_normal(
            (self.N, self.output_dim, self.input_dim)
        )

        # Collect 2N rollouts of horizon H in parallel
        with mp.Pool(self.num_cores) as pool:
            pos_reward_lst = []
            pos_state_lst = []
            pos_ret = pool.starmap(
                roll_out, 
                [
                    (
                        self.env_name, 
                        self.weight + self.nu * directions[i], 
                        self.normalizer, 
                        self.seed + i
                    ) for i in range(self.N)
                ]
            )
            for item in pos_ret:
                pos_reward_lst.append(item[0])
                pos_state_lst.append(item[1])

            neg_reward_lst = []
            neg_state_lst = []
            neg_ret = pool.starmap(
                roll_out, 
                [
                    (
                        self.env_name, 
                        self.weight - self.nu * directions[i], 
                        self.normalizer, 
                        self.seed + i
                    ) for i in range(self.N)
                ]
            )
            for item in neg_ret:
                neg_reward_lst.append(item[0])
                neg_state_lst.append(item[1])
            
        
        pos_rewards = np.array(pos_reward_lst)
        neg_rewards = np.array(neg_reward_lst)

        differences = pos_rewards - neg_rewards

        std = np.std(np.concatenate([pos_rewards, neg_rewards]))

        # Update weight
        self.weight += self.alpha / std * np.mean(
            differences[:, None, None] * directions, axis=0
        )

        # Update normalizer
        for s_lst in pos_state_lst:
            for s in s_lst:
                self.normalizer.observe(s)
        for s_lst in neg_state_lst:
            for s in s_lst:
                self.normalizer.observe(s)

        rendering=False
        if num_iters % 50 == 0:
            rendering=True
        reward, _ = roll_out(
            env_name=self.env_name, 
            weight=self.weight, 
            normalizer=self.normalizer, 
            seed=num_iters,
            rendering=rendering
        )

        return reward, std


        