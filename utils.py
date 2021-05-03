import tensorflow.compat.v1 as tf

"""from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tf_agents.policies.random_tf_policy as random_tf_policy
from tf_agents.specs import tensor_spec


import tf_agents.agents.tf_agent as tf_agent
from tf_agents.drivers import dynamic_step_driver


from tf_agents.trajectories import policy_step"""
from tf_agents.trajectories import trajectory


def compute_avg_return(env, policy, num_episodes=10):
    """
    :param env: tf_agents tensorized environment.
    :param policy: an agent.policy or agent.collect_policy object.
    :param num_episodes: number of collections to average over.
    :return: reward that the policy acting in the environment achieved, averaged over num_episodes runs.
    """
    total_return = 0.0
    time_step = env.current_time_step()

    for _ in range(num_episodes):
        episode_return = 0.0
        # environment might be passed in already reset.
        if time_step.is_last():
            time_step = env.reset()
        policy_state = policy.get_initial_state(env.batch_size)
        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            time_step = env.step(action_step.action)
            policy_state = action_step.state
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_episode(env, policy, replay_buffer, num_episodes):
    """
    Collects one or more trajectories and puts them into the replay buffer.
    :param env: tf_agents tensorized environment.
    :param policy: an agent.collect_policy object.
    :param replay_buffer: a tf_agents replay buffer to keep the collected transitions in.
    :param num_episodes: number of episodes to collect at once.
    :return: None.
    """
    episode_counter = 0
    if env.current_time_step().is_last():
        env.reset()
    policy_state = policy.get_initial_state(env.batch_size)

    while episode_counter < num_episodes:
        time_step = env.current_time_step()

        action_step = policy.action(time_step, policy_state)
        next_time_step = env.step(action_step.action)
        policy_state = action_step.state
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1
            policy_state = policy.get_initial_state(env.batch_size)


def log_ppo_loss(loss):
    """
    Prints PPO loss information.
    :param loss: loss object from a PPOAgent.train(...) call.
    :return: None.
    """
    print("policy_gradient_loss =", loss.extra.policy_gradient_loss.numpy())
    print("value_estimation_loss =", loss.extra.value_estimation_loss.numpy())
    print("l2_regularization_loss =", loss.extra.l2_regularization_loss.numpy())
    print("entropy_regularization_loss =", loss.extra.entropy_regularization_loss.numpy())
    print("kl_penalty_loss =", loss.extra.kl_penalty_loss.numpy())
