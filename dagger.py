import gym
import numpy as np
import tempfile

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer


ENV = 'Lunar Lander' # 'Lunar Lander' or 'Bipedal Walker'
TRAIN = False 
N = 10 # Number of initial expert demonstrations 
T = 1000 # Number of timesteps per trajectory
rng = np.random.default_rng(0)


if ENV == 'Lunar Lander':
    env = make_vec_env("LunarLander-v2", n_envs=1)
    checkpoint = load_from_hub("araffin/ppo-LunarLander-v2", "ppo-LunarLander-v2.zip")
    expert = PPO.load(checkpoint)
elif ENV == 'Bipedal Walker':
    env = make_vec_env("BipedalWalker-v3", n_envs=1)
    checkpoint = load_from_hub()

if TRAIN:
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
        )

        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )

        dagger_trainer.train(2000)
    