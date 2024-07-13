import numpy as np

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
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
    

"""
Implementation of ARS V2-t.
Parameters: 
 - alpha: step-size
 - N: number of directions sampled per iteration
 - nu: standard deviation of the exploration nosie
 - b: number of top-performing directions to use
"""
class ARS():
    def __init__(self, input_dim, output_dim, alpha, N, nu, b, seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.N = N
        self.nu = nu
        self.b = b
        self.seed = seed

        self.normalizer = Normalizer(input_dim)
        self.weights = np.zeros((self.output_dim, self.input_dim))

        np.random.seed(self.seed)

    def roll_out(self, w, env, training=True):
        sigmoid = lambda x : x / np.sqrt(1 + x ** 2)

        env.action_space.seed(self.seed)
        s, _ = env.reset(seed=self.seed)
        total_reward = 0
        steps = 0
        H = 2000

        while steps < H:
            if training:
                self.normalizer.observe(s)
            s = self.normalizer.normalize(s)
            a = sigmoid(w.dot(s)) # restrict the action to be in the range [-1, 1]
            s, r, terminated, truncated, _ = env.step(a)
            total_reward += r
            steps += 1
            if terminated or truncated:
                break

        return total_reward

    def train_one_iter(self, env):
        directions = np.random.standard_normal(
            (self.N, self.output_dim, self.input_dim)
        )
        
        max_rewards = np.zeros(self.N)
        pos_rewards = np.zeros(self.N)
        neg_rewards = np.zeros(self.N)

        for i in range(self.N):
            pos_w = self.weights + self.nu * directions[i]
            pos_rewards[i] = self.roll_out(pos_w, env)

            neg_w = self.weights - self.nu * directions[i]
            neg_rewards[i] = self.roll_out(neg_w, env)

            max_rewards[i] = max(pos_rewards[i], neg_rewards[i])

        differences = pos_rewards - neg_rewards

        idx = np.argsort(max_rewards)[-self.b:]
        std = np.std(np.concatenate([pos_rewards[idx], neg_rewards[idx]]))

        self.weights += self.alpha / std * np.mean(
            differences[idx][:, None, None] * directions[idx], axis=0
        )

        return self.roll_out(self.weights, env, training=False)


        