from model import ARS
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    ars = ARS(input_dim=24, output_dim=4, alpha=0.02, N=30, nu=0.02, b=15, seed=42)
    
    num_iters = 500
    total_rewards = []
    pbar = tqdm(range(num_iters), desc='Training ARS: ')

    for i in pbar:
        reward = ars.train_one_iter(env)
        total_rewards.append(reward)
        pbar.set_postfix({'Reward': reward})

    print('Training complete.')
    print('Weights: \n', ars.weights)
    normalizer_params = {
        'mean' : ars.normalizer.mean,
        'mean_diff' : ars.normalizer.mean_diff,
    }
    print('Normalizer params: \n', normalizer_params)

    plt.plot(total_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.title('ARS V2-t on BipedalWalker-v3')
    plt.show()

    # Save rewards
    with open('ars_v2t_bipedalwalker.pkl', 'wb') as f:
        pickle.dump(total_rewards, f)

    # Save weights
    with open('ars_v2t_bipedalwalker_weights.pkl', 'wb') as f:
        pickle.dump(ars.weights, f)

    # Save normalizer
    with open('ars_v2t_bipedalwalker_normalizer.pkl', 'wb') as f:
        pickle.dump(normalizer_params, f)
    


    