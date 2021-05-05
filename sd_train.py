#!/usr/bin/env python3

import os
import numpy as np

import gym
from stable_baselines3 import PPO
from sacred import Experiment, observers

from wrappers import BackwardsResetWrapper


sd_experiment = Experiment("single_demonstration")
observer = observers.FileStorageObserver('results/sd_results')
sd_experiment.observers.append(observer)

TASK_INFO = {'space': {'name': 'SpaceInvadersDeterministic-v4',
                       'traj': 'data/space_action_seq.json'},
             'spaceram': {'name': 'SpaceInvaders-ramDeterministic-v4',
                       'traj': 'data/space_action_seq.json'}
             }


@sd_experiment.config
def base_config():
    task = 'space'
    # 200,000 is not enough to learn the full task -- this is model-free RL
    #   however, it is enough to see signs of life. You should learn the last
    #   ~10% with this.
    total_timesteps = 200000
    eval_freq = 10000
    eval_episodes = 3
    # See BackwardsResetWrapper for param descriptions
    history_len = 5
    reset_step = 5
    advance_limit = 0.8
    reward_thresh = 1200 # From the specified rollout

    policy = 'MlpPolicy' if task == 'spaceram' else 'CnnPolicy'

    save_policy = True


@sd_experiment.capture
def sd_reset_params(history_len, reset_step, advance_limit, reward_thresh):
    return {'history_len': history_len,
            'reset_step': reset_step,
            'advance_limit': advance_limit,
            'reward_thresh': reward_thresh}


@sd_experiment.automain
def experiment_main(task, total_timesteps, eval_freq, eval_episodes,
                    policy, save_policy):
    # Use a hard navigation task with sparse reward. Demo trajectory 
    #   requires 266 actions to get first reward.
    task_details = TASK_INFO[task]

    env = gym.make(task_details['name'])
    reset_env = BackwardsResetWrapper(env, task_details['traj'],
                                      **sd_reset_params())
    
    eval_env = gym.make(task_details['name'])


    model = PPO(policy, reset_env, verbose=1)

    model.learn(total_timesteps=total_timesteps,
                eval_env=eval_env, eval_freq=eval_freq, 
                n_eval_episodes=eval_episodes)

    


    if save_policy:
        model.save(os.path.join(observer.dir, 'policy'))

    env.close()