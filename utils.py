from wrappers import BackwardsResetWrapper

def compute_avg_return(env, model, eval_episodes):
    rollouts = []
    for _ in range(eval_episodes)
        obs = env.reset()
        done = False
        r = 0
        while not done:
            obs, rew, done, info = env.step(model.predict(obs))
            r += rew

        rollouts += r

    return sum(rollouts) / len(rollouts)


def sd_eval(model, eval_env, cur_reset, step_size=5, eval_episodes=3, experiment=None):
    """ Used to eval learning from a single demonstration:
        Evaluates the model on 
    """
    for i in range(0, cur_reset, step_size):
        env = BackwardsResetWrapper(eval_env)
        reset_env.reset_point = i

        avg_return = compute_avg_return(train_env, model, eval_episodes)
        print('iteration = {0}: Average Return = {1} when starting {2} steps from the end'.format(iteration, avg_return, i))
        if experiment is not None:
            experiment.log_scalar("{0}_step_return".format(i), avg_return)
