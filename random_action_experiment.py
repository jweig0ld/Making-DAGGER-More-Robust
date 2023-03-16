import gym
import numpy as np
import tempfile
import os
import torch as th
import random
import matplotlib.pyplot as plt

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import bc 
from imitation.algorithms.bc import reconstruct_policy

from torch import nn

from copy import deepcopy


def compute_policy_statistics(model, p_random):
    n_eval_episodes = 50 # Same as is used in the imitation implementation
    all_rewards = [] # Assumes that this is a list of scalars NOT arrays

    for episode in range(n_eval_episodes):
        env = make_vec_env("LunarLander-v2", n_envs=1)
        obs = env.reset()

        episode_rewards = []
        done = False
        while not done:
            action, value, log_prob = model(th.from_numpy(obs))

            # Take a random action with probability `p_random`
            action = th.tensor([env.action_space.sample()]) if random.random() < p_random else action

            obs, reward, done, info = env.step(action.numpy())
            episode_rewards.append(reward.item())
        
        all_rewards.append(episode_rewards)
    
    mean_episode_length = sum([len(ep_rewards) for ep_rewards in all_rewards]) / n_eval_episodes
    mean_episode_reward = sum([sum(ep_rewards) for ep_rewards in all_rewards]) / n_eval_episodes
    
    max_length = max(len(lst) for lst in all_rewards)
    padded_all_rewards = [lst + [0]*(max_length - len(lst)) for lst in all_rewards]

    all_rewards_th = th.tensor(padded_all_rewards)
    episode_rewards_th = th.sum(all_rewards_th, dim=1)
    std_episode_reward = th.std(episode_rewards_th)

    # Proportion of episodes resulting in a crash
    crash_count = 0
    for ep_rewards in all_rewards:
        for r in ep_rewards:
            if r <= -100:
                crash_count += 1
    
    crash_percentage = crash_count / n_eval_episodes

    return {'mean_episode_length': mean_episode_length, 
            'mean_episode_reward': mean_episode_reward, 
            'std_episode_reward': std_episode_reward, 
            'crash_percentage': crash_percentage}


def _get_statistics(policy_paths, mts, p_random):
    stats = {}
    for i in range(len(policy_paths)):
        policy = reconstruct_policy(policy_paths[i])
        results_dict = compute_policy_statistics(policy, p_random)
        stats[mts[i]] = results_dict
    return stats


def plot_lines(mts, dagger_data, k2_data, k3_data, xlabel, ylabel, title, filename=None):
    # Plot the lines
    plt.plot(mts, dagger_data, label='DAGGER')
    plt.plot(mts, k2_data, label='EDAGGER K=2')
    plt.plot(mts, k3_data, label='EDAGGER K=3')

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add legend
    plt.legend()

    # Save the plot if filename is supplied
    if filename:
        plt.savefig(filename)

    # Display the plot
    plt.show()


def plot_episode_lengths(dagger_stats, k2_stats, k3_stats, mts, filename):
    dagger_ep_lengths = [dagger_stats[mt]['mean_episode_length'] for mt in mts]
    k2_ep_lengths = [k2_stats[mt]['mean_episode_length'] for mt in mts]
    k3_ep_lengths = [k3_stats[mt]['mean_episode_length'] for mt in mts]

    plot_lines(mts, 
               dagger_data=dagger_ep_lengths, 
               k2_data=k2_ep_lengths, 
               k3_data=k3_ep_lengths,
               xlabel='Dataset Size',
               ylabel='Mean Episode Length',
               title='',
               filename=filename)
    

def plot_crash_percentage(dagger_stats, k2_stats, k3_stats, mts, filename):
    dagger_ep_lengths = [dagger_stats[mt]['crash_percentage'] for mt in mts]
    k2_ep_lengths = [k2_stats[mt]['crash_percentage'] for mt in mts]
    k3_ep_lengths = [k3_stats[mt]['crash_percentage'] for mt in mts]

    plot_lines(mts, 
               dagger_data=dagger_ep_lengths, 
               k2_data=k2_ep_lengths, 
               k3_data=k3_ep_lengths,
               xlabel='Dataset Size',
               ylabel='Proportion of Episodes Crashing',
               title='',
               filename=filename)
    

def plot_mean_rewards(mts, dagger_stats, k2_stats, k3_stats, filename=None, xlabel=None, ylabel=None):
    dagger_mu = np.array([dagger_stats[mt]['mean_episode_reward'] for mt in mts])
    dagger_std = np.array([dagger_stats[mt]['std_episode_reward'] for mt in mts])

    k2_mu = np.array([k2_stats[mt]['mean_episode_reward'] for mt in mts])
    k2_std = np.array([k2_stats[mt]['std_episode_reward'] for mt in mts])

    k3_mu = np.array([k3_stats[mt]['mean_episode_reward'] for mt in mts])
    k3_std = np.array([k3_stats[mt]['std_episode_reward'] for mt in mts])

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the lines and error bars
    ax.plot(mts, dagger_mu, label='DAGGER')
    ax.fill_between(mts, dagger_mu - dagger_std, dagger_mu + dagger_std, alpha=0.2)

    ax.plot(mts, k2_mu, label='EDAGGER K=2')
    ax.fill_between(mts, k2_mu - k2_std, k2_mu + k2_std, alpha=0.2)

    ax.plot(mts, k3_mu, label='EDAGGER K=3')
    ax.fill_between(mts, k3_mu - k3_std, k3_mu + k3_std, alpha=0.2)

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('')

    # Add legend
    ax.legend()

    # Save the plot if filename is supplied
    if filename:
        plt.savefig(filename)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    policy_paths = []
    mts = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7500, 10000]
    # mts = []
    dagger_policy_paths = ['dagger_mt_' + str(mt) + '_policy.pt' for mt in mts]
    k3_policy_paths = ['double_fix_exploratory_dagger_k3_mt_' + str(mt) + '_policy.pt' for mt in mts]
    k2_policy_paths = ['double_fix_exploratory_dagger_k2_mt_' + str(mt) + '_policy.pt' for mt in mts]

    ps = [0., 0.05, 0.1, 0.15, 0.2]
    for p_random in ps:

        print(f'Calculating dagger stats for p={p_random}...')
        dagger_stats = _get_statistics(dagger_policy_paths, mts, p_random)
        print(f'Calculating K2 stats for p={p_random}...')
        k2_stats = _get_statistics(k2_policy_paths, mts, p_random)
        print(f'Calculating K3 stats for p={p_random}...')
        k3_stats = _get_statistics(k3_policy_paths, mts, p_random)

        plot_episode_lengths(dagger_stats, k2_stats, k3_stats, mts, filename='A_Mean_Episode_Length_P=' + str(p_random) +'.png')
        plot_crash_percentage(dagger_stats, k2_stats, k3_stats, mts, filename='A_Crash_Percentage_P=' + str(p_random) +'.png')
        plot_mean_rewards(mts, dagger_stats, k2_stats, k3_stats, xlabel='Dataset Size', ylabel='Mean Episode Reward', filename='A_Mean_Rewards_P='+str(p_random)+'.png')