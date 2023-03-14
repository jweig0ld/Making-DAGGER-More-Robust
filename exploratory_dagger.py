import gym
import os
import random
import torch
import numpy as np
import lightning.pytorch as pl

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

from dataset import ExploratoryDaggerDataset


"""
TODO:

1. (DONE) Load the pre-trained weights from Lunar Lander for the expert.
2. (DONE) Find out how long trajectories are in the other implementation.
3. (DONE) Find out how many initial trajectories that the other one starts with (ANS = 0).
4. (DONE) Run inference on the expert agent to generate the same initial trajectories (NO NEED TO).
5. (DONE) Find out how many trajectories DAGGER generates before a training step.
6. (DONE) Find out what their ultimate training termination condition is.

1. (DONE) For every observation, select the top-k highest actions.
2. (DONE) Create k copies of the environment, and simulate taking different actions in each.
3. (DONE) Find out what the RND paper uses to determine the greatest difference in output (MSE).
4. (DONE) Run the resulting k new observations through the frozen and unfrozen networks.
5. (DONE) Find out what loss the RND paper uses.
6. (DONE) Take the action which results in the highest difference.
7. (DONE) Find out what the beta schedule that they use for the expert is (15 rounds).
8. (DONE) Find out approximately how many training steps 15 rounds is. (at least 500 timesteps). 500 * 15 = 7500.
9. (DONE) Implement the beta schedule s.t. if a random number is less than the beta, we query the agent.
10. (DONE) Accumulate the action taken in the global dataset and query the expert for the correct one.
11. (DONE) Find out how to write a trivial BC training step/loop using PyTorch Lightning.
12. (DONE) Find out what loss they were using for their network (entropy-regularised NLL loss).
13. (DONE) Find out how to write a naive PyTorch DataLoader.
14. (DONE) Write a PyTorch Dataset class.
15. (DONE) Find out how I should perform the BC training step.
16. (DONE) Implement a DataLoader for the Dataset.
17. (DONE) Perform a BC training step through the entire dataset.
"""

# CONFIG
ENV = 'Lunar Lander' # 'Lunar Lander' or 'Bipedal Walker'
TRAIN = True 
rng = np.random.default_rng(0)
CHECKPOINT_RATE = 1000
LOGDIR = 'logs'
version = 9

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

        self.frozen_net = type(self.random_net)(in_dim, h_dim, out_dim)
        self.frozen_net.load_state_dict(self.random_net.state_dict())

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
        y_hat = self(x)
        loss = F.nll_loss(y, y_hat)
        # # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
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
                              in_dim=env.observation_space.n,
                              h_dim=H_DIM,
                              out_dim=env.action_space.n)
    trainer = pl.Trainer()

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

            # Generate a trajectory of data
            while not done:
                action = None
                if random.random() < model.beta_schedule(timestep=total_timesteps, max_timesteps=MAX_TIMESTEPS):
                    # Query expert for action
                    action = expert(obs)
                else:
                    probs = model(obs)
                    top_k_actions = torch.topk(probs, model.k)[1] # Argmax returns the actions

                    # Simulate taking the top-k actions in independent environments
                    env_copies, fut_obs = [deepcopy(env) for _ in range(K)], []
                    for i in range(K):
                        # Get the observations after one future action
                        obs_prime, _, _, _ = env_copies[i].step(top_k_actions[i])
                        fut_obs.append(obs_prime)
                    
                    # Pass the observations through the frozen and unfrozen networks and pick
                    # the one with the highest difference. 
                    frozen_outs = torch.tensor([model.frozen_net(fut) for fut in fut_obs], requires_grad=False)
                    random_outs = torch.tensor([model.random_net(fut) for fut in fut_obs])
                    mse = rnd_loss(random_outs, frozen_outs) # We are expecting this to be (k x 1)
                    action = torch.argmax(mse, dim=0)
                    
                obs, reward, done, info = env.step(action)
                epoch_timesteps += 1
                total_timesteps += 1
                epoch_observations.append(obs)

            epoch_episodes += 1

        # After sufficient training data has been collected, query the expert on the observations
        # during the dataset aggregation period
        epoch_expert_actions = expert(epoch_observations) # (timesteps x action_space)
        dataset.aggregate(epoch_observations, epoch_expert_actions)
        
        epoch_observations = []

        ############ BC Training Step ############
        model = ExploratoryDagger(k=K, 
                              in_dim=env.observation_space.n,
                              h_dim=H_DIM,
                              out_dim=env.action_space.n)
        
        dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
        trainer.fit(model, dataloader)
        
