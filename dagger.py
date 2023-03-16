import gym
import numpy as np
import tempfile
import os

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.logger import make_output_format, configure

from torch import nn

from copy import deepcopy


ENV = 'Lunar Lander' # 'Lunar Lander' or 'Bipedal Walker'
TRAIN = True 
rng = np.random.default_rng(0)
CHECKPOINT_RATE = 1000
LOGDIR = 'logs'

# DAGGER CONFIG
H_DIM = 32
OUT_DIM = 1
MAX_TIMESTEPS = 10000


if ENV == 'Lunar Lander':
    env = make_vec_env("LunarLander-v2", n_envs=1)
    checkpoint = load_from_hub("araffin/ppo-LunarLander-v2", "ppo-LunarLander-v2.zip")
    expert = PPO.load(checkpoint)

if TRAIN:
    path = "triple_fix_exploratory_dagger_k2_mt_" + str(MAX_TIMESTEPS)
    # logger = make_output_format('csv', LOGDIR)
    logger = configure('triple_fix_exploratory_dagger_k2_logs_mt_' + str(MAX_TIMESTEPS), ['csv'])

    try:
      os.mkdir(path)
    except OSError as error:
      print(error)

    online_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], H_DIM),
            nn.ReLU(),
            nn.Linear(H_DIM, H_DIM),
            nn.ReLU(),
            nn.Linear(H_DIM, OUT_DIM),
    )

    frozen_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], H_DIM),
            nn.ReLU(),
            nn.Linear(H_DIM, H_DIM),
            nn.ReLU(),
            nn.Linear(H_DIM, OUT_DIM),
    )

    for param in frozen_net.parameters():
        param.requires_grad = False

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        online_net=online_net,
        frozen_net=frozen_net,
        rng=rng,
        custom_logger=logger,
    )

    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=path,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng
    )

    print("Commencing Training...")
    dagger_trainer.train(MAX_TIMESTEPS, bc_train_kwargs={'log_interval': 50})
    # dagger_trainer.save_trainer()
    dagger_trainer.save_policy(path+"_policy.pt")