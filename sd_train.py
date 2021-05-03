#!/usr/bin/env python3

import gym

import os
import numpy as np
import tensorflow as tf

# pip install tensorflow, tf-agents, gym[atari], sacred...

from sacred import Experiment, observers

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.networks import actor_distribution_rnn_network, value_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver

from utils import collect_episode, compute_avg_return, log_ppo_loss
from wrappers import BackwardsResetWrapper

tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()

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
    num_iterations = 400
    episodes_per_iter = 1
    
    # How often to log training metrics
    log_iter = 8
    # How often to eval and save policy
    eval_iter = 80

    lr = 1e-3
    # Policy arguments
    actor_conv_layer = [(4, 2, 2)]
    actor_fc_layer = (100, 100,)
    value_conv_layer = [(4, 2, 2)]
    value_fc_layer = (100, 100,)
    use_rnns = False
    lstm_size = 128
    replay_buffer_max_length = 1000
    eval_episodes = 3
    # See BackwardsResetWrapper for param descriptions
    history_len = 5
    reset_step = 3
    advance_limit = 0.8
    reward_thresh = 600
    save_policy = True



@sd_experiment.capture
def sd_reset_params(history_len, reset_step, advance_limit, reward_thresh):
    return {'history_len': history_len,
            'reset_step': reset_step,
            'advance_limit': advance_limit,
            'reward_thresh': reward_thresh}


@sd_experiment.capture
def make_actor_net(train_env, actor_conv_layer, actor_fc_layer):
    return actor_distribution_network.ActorDistributionNetwork(train_env.observation_spec(),
                                                               train_env.action_spec(),
                                                               conv_layer_params=actor_conv_layer,
                                                               fc_layer_params=actor_fc_layer)


@sd_experiment.capture
def make_rnn_actor_net(train_env, actor_conv_layer, actor_fc_layer, lstm_size):
    return actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        conv_layer_params=actor_conv_layer,
        input_fc_layer_params=actor_fc_layer,
        lstm_size=(lstm_size,),
        output_fc_layer_params=actor_fc_layer,
    )


@sd_experiment.capture
def make_value_net(train_env, value_conv_layer, value_fc_layer):
    return value_network.ValueNetwork(train_env.observation_spec(),
                                      conv_layer_params=value_conv_layer,
                                      fc_layer_params=value_fc_layer)


@sd_experiment.capture
def make_rnn_value_net(train_env, value_conv_layer, value_fc_layer, lstm_size):
    return value_rnn_network.ValueRnnNetwork(
        train_env.observation_spec(),
        conv_layer_params=value_conv_layer,
        input_fc_layer_params=value_fc_layer,
        lstm_size=(lstm_size,),
        output_fc_layer_params=value_fc_layer,
    )


@sd_experiment.capture
def make_optimizer(lr):
    return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)


@sd_experiment.capture
def make_replay_buffer(agent, train_env, replay_buffer_max_length):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                          batch_size=train_env.batch_size,
                                                          max_length=replay_buffer_max_length)


@sd_experiment.automain
def experiment_main(task, use_rnns, num_iterations, episodes_per_iter,
                    log_iter, eval_iter, eval_episodes, save_policy):
    # Use a hard navigation task with sparse reward. Demo trajectory requires 266 actions to get first reward.
    task_details = TASK_INFO[task]

    env = gym.make(task_details['name'])
    reset_env = BackwardsResetWrapper(env, task_details['traj'],
                                      **sd_reset_params())
    py_env = gym_wrapper.GymWrapper(reset_env)
    # use a tensorized environment
    train_env = tf_py_environment.TFPyEnvironment(py_env)

    optimizer = make_optimizer()
    if use_rnns:
        actor_net = make_rnn_actor_net(train_env)
        value_net = make_rnn_value_net(train_env)
    else:
        actor_net = make_actor_net(train_env)
        value_net = make_value_net(train_env)
    train_step_counter = tf.Variable(0)

    agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        debug_summaries=True,
        train_step_counter=train_step_counter,
    )

    agent.initialize()

    replay_buffer = make_replay_buffer(agent, train_env)

    if save_policy:
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    for iteration in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_episode(train_env, agent.collect_policy, replay_buffer, episodes_per_iter)

        # Sample a batch of data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()

        total_loss = agent.train(experience)
        replay_buffer.clear()
        train_loss = total_loss.loss

        if iteration % log_iter == 0:
            print('iteration = {0}: loss = {1}'.format(iteration, train_loss))
            sd_experiment.log_scalar("training loss", train_loss.numpy())
            print('current starting point: {0}/{1}'.format(reset_env.trajectory_len - reset_env.reset_point,
                                                           reset_env.trajectory_len))
            sd_experiment.log_scalar("learned steps", reset_env.reset_point)

        if iteration % eval_iter == 0:
            recent_rewards = reset_env.recent_rewards
            reset_point = reset_env.reset_point

            for i in range(0, reset_env.reset_point, 5):
                reset_env.recent_rewards = [0] * reset_env.history_len
                reset_env.reset_point = i

                avg_return = compute_avg_return(train_env, agent.policy, eval_episodes)
                print('iteration = {0}: Average Return = {1} when starting {2} steps from the end'.format(iteration, avg_return, i))
                sd_experiment.log_scalar("{0}_step_return".format(i), avg_return)

            reset_env.recent_rewards = recent_rewards
            reset_env.reset_point = reset_point

            if save_policy:
                tf_policy_saver.save(os.path.join(observer.dir, 'policy'))
                print("Saved policy.")

    if save_policy and iteration % eval_iter != 0:
        tf_policy_saver.save(os.path.join(observer.dir, 'policy'))

    env.close()