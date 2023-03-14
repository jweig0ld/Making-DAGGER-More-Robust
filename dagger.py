import gym
import os
import random
import torch
import numpy as np
import pytorch_lightning as pl

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.logger import make_output_format, configure
from imitation.policies.base import FeedForward32Policy

from torch import optim, nn, utils, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from copy import deepcopy


def compute_model_statistics(model: pl.LightningModule):
    n_eval_episodes = 5 # Same as is used in the imitation implementation
    all_rewards = []
    for episode in range(n_eval_episodes):
        env = make_vec_env("LunarLander-v2", n_envs=1)
        obs = env.reset()

        episode_rewards = []
        done = False
        while not done:
            action, value, log_prob = model(torch.from_numpy(obs))
            obs, reward, done, info = env.step(action.numpy())
            episode_rewards.append(reward.item())
        
        all_rewards.append(episode_rewards)
    
    mean_episode_length = sum([len(ep_rewards) for ep_rewards in all_rewards]) / n_eval_episodes
    mean_episode_reward = sum([sum(ep_rewards) for ep_rewards in all_rewards]) / n_eval_episodes
    
    max_length = max(len(lst) for lst in all_rewards)
    padded_all_rewards = [lst + [0]*(max_length - len(lst)) for lst in all_rewards]

    all_rewards_th = torch.tensor(padded_all_rewards)
    episode_rewards_th = torch.sum(all_rewards_th, dim=1)
    std_episode_reward = torch.std(episode_rewards_th)

    return mean_episode_length, mean_episode_reward, std_episode_reward


# CONFIG
ENV = 'Lunar Lander' # 'Lunar Lander' or 'Bipedal Walker'
TRAIN = True 
rng = np.random.default_rng(0)
CHECKPOINT_RATE = 1000
LOGDIR = 'logs'
version = 9
RND = False

# DAGGER PARAMS
BATCH_SIZE = 32
MIN_EPISODES = 3 # Minimum number of episodes completed in dataset aggregation phase
MIN_TIMESTEPS = 500 # Minimum number of timesteps completed in dataset aggregation
MAX_TIMESTEPS = 5000 # Lower bound on total training termination condition
K = 3 # Number of actions that we simulate taking
BC_EPOCHS = 4 # Number of BC epochs performed in a training step before returning to dataset agg

# RND PARAMS
H_DIM = 32


class ExploratoryDagger(pl.LightningModule):
    def __init__(self, k, in_dim, h_dim, out_dim, observation_space, action_space):
        super().__init__()

        # Number of actions to simulate taking
        self.k = k
        
        # Random Network Distillation
        self.random_net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
            nn.Softmax()
        )

        self.frozen_net = deepcopy(self.random_net)

        # Duplicating DAGGER policy architecture
        self.policy = FeedForward32Policy(
            observation_space=observation_space,
            action_space=action_space,
            # Set lr_schedule to max value to force error if policy.optimizer
            # is used by mistake (should use self.optimizer instead).
            lr_schedule=lambda _: torch.finfo(torch.float32).max,
        )
        # NOTE: The dimension of the output of self.policy MUST be the number
        # of actions in the action space.

    def forward(self, x):
        return self.policy(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # actions, values, log_probs = self(x)
        # print(f'X Batch: {x.shape}')
        log_probs = model.policy.get_distribution(x).distribution.logits
        loss = F.nll_loss(log_probs, y)
        # # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)

        # loss += F.mse(self.random_net(x), self.frozen_net(x))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def beta_schedule(self, timestep, max_timesteps):
        # For a given timestep <= max_timestep, return a linearly decreasing beta.
        return (max_timesteps - timestep) / max_timesteps
   

if ENV == 'Lunar Lander':
    env = make_vec_env("LunarLander-v2", n_envs=1)
    checkpoint = load_from_hub("araffin/ppo-LunarLander-v2", "ppo-LunarLander-v2.zip")
    expert = PPO.load(checkpoint)
elif ENV == 'Bipedal Walker':
    env = make_vec_env("BipedalWalker-v3", n_envs=1)
    checkpoint = load_from_hub()
    expert = None


if TRAIN:
    print("Commencing Training...")

    model = ExploratoryDagger(k=K, 
                              in_dim=env.observation_space.shape[0],
                              h_dim=H_DIM,
                              out_dim=1,
                              observation_space=env.observation_space,
                              action_space=env.action_space)

    rnd_loss = nn.MSELoss(reduction='none')
    dataset = ExploratoryDaggerDataset()

    total_timesteps = 0 # Accumulator for total number of env timesteps performed
    while total_timesteps < MAX_TIMESTEPS:

        ############ Dataset Aggregation ############

        # Dataset aggregation – generate at least MIN_EPISODES episodes and 
        # MIN_TIMESTEPS timesteps worth of training data before performing
        # training steps.
        epoch_episodes, epoch_timesteps = 0, 0
        epoch_observations = []
        while epoch_episodes < MIN_EPISODES and epoch_timesteps < MIN_TIMESTEPS:
            env = make_vec_env("LunarLander-v2", n_envs=1)
            obs = env.reset()
            print(f'Obs: {obs}')

            done = False
            episode_actions = []
            episode_reward = 0

            # Generate a trajectory of data
            while not done:
                action = None
                if random.random() < model.beta_schedule(timestep=total_timesteps, max_timesteps=MAX_TIMESTEPS):
                    # Query expert for action
                    # print('Queried the expert for an action.')
                    action, _ = expert.predict(obs)
                else:
                    # print('Queried the agent for an action.')
                    obs = torch.tensor(obs)

                    if RND: 
                        actions = torch.tensor([0., 1., 2., 3.])
                        log_probs = model.policy.get_distribution(obs).log_prob(actions)
                        top_k_actions = torch.topk(log_probs, model.k)[1] # Argmax returns the actions
                        
                        # Simulate taking the top-k actions in independent environments
                        env_copies, fut_obs = [make_vec_env("LunarLander-v2", n_envs=1) for _ in range(K)], []
                        for i in range(K):
                            env_copies[i].reset()

                        for past_action in episode_actions:
                            # Run the copied environments forward to the current state
                            for i in range(K):
                                env_copies[i].step(past_action)

                        for i in range(K):
                            # Get the observations after one future action
                            candidate_action = np.array([top_k_actions[i]])
                            obs_prime, _, _, _ = env_copies[i].step(candidate_action)
                            fut_obs.append(torch.from_numpy(obs_prime))
                        
                        # Pass the observations through the frozen and unfrozen networks and pick
                        # the one with the highest difference. 
                        frozen_outs = torch.tensor([model.frozen_net(fut) for fut in fut_obs], requires_grad=False)
                        random_outs = torch.tensor([model.random_net(fut) for fut in fut_obs])
                        mse = rnd_loss(random_outs, frozen_outs) # We are expecting this to be (k x 1)
                        action = np.array([torch.argmax(mse, dim=0)])
                    else:
                        action, value, log_prob = model(obs)
                        action = action.numpy()
                    
                
                episode_actions.append(action)

                # print(f'Action: {action}.')
                obs, reward, done, info = env.step(action)
                episode_reward += reward

                epoch_timesteps += 1
                total_timesteps += 1
                epoch_observations.append(obs)

            epoch_episodes += 1

        # After sufficient training data has been collected, query the expert on the observations
        # during the dataset aggregation period
        # epoch_expert_actions, _ = expert.predict(epoch_observations) # (timesteps x action_space)
        _expert_outs = [expert.predict(observation) for observation in epoch_observations]
        epoch_expert_actions = [action for (action, _) in _expert_outs]
        dataset.aggregate(epoch_observations, epoch_expert_actions)
        
        epoch_observations = []

        ############ BC Training Step ############
        model = ExploratoryDagger(k=K, 
                              in_dim=env.observation_space.shape[0],
                              h_dim=H_DIM,
                              out_dim=1,
                              observation_space=env.observation_space,
                              action_space=env.action_space)
        
        
        dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
        trainer = pl.Trainer(max_epochs=BC_EPOCHS)
        trainer.fit(model, dataloader)

        ############ Validation & Statistics ############
        mean_episode_length, mean_episode_reward, std_episode_reward = compute_model_statistics(model)
        print(f'Mean Episode Length: {mean_episode_length}. Mean Episode Reward: {mean_episode_reward}. Std Episode Reward: {std_episode_reward}.')

        print(f'BC Step Complete. Total Timesteps: {total_timesteps}.')
        
print('Training Complete.')