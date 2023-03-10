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


ENV = 'Lunar Lander' # 'Lunar Lander' or 'Bipedal Walker'
TRAIN = True 
N = 10 # Number of initial expert demonstrations 
T = 1000 # Number of timesteps per trajectory
rng = np.random.default_rng(0)
CHECKPOINT_RATE = 1000
LOGDIR = 'logs'


if ENV == 'Lunar Lander':
    env = make_vec_env("LunarLander-v2", n_envs=1)
    checkpoint = load_from_hub("araffin/ppo-LunarLander-v2", "ppo-LunarLander-v2.zip")
    expert = PPO.load(checkpoint)
elif ENV == 'Bipedal Walker':
    env = make_vec_env("BipedalWalker-v3", n_envs=1)
    checkpoint = load_from_hub()

if TRAIN:
    version = 9
    path = "dagger_example_" + str(version)
    # logger = make_output_format('csv', LOGDIR)
    logger = configure('logs' + str(version), ['csv'])

    try:
      os.mkdir(path)
    except OSError as error:
      print(error)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
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
    dagger_trainer.train(5000, bc_train_kwargs={'log_interval': 50})
    # dagger_trainer.save_trainer()
    dagger_trainer.save_policy(path+"_policy.pt")