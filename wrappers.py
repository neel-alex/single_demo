import random
import json

from gym import Wrapper


class BackwardsResetWrapper(Wrapper):
    def __init__(self, env, traj_fname, history_len=5, reset_step=3, 
                 advance_limit=0.8, reward_thresh=1):
        """
        Implements https://arxiv.org/pdf/1812.03381.pdf (single-threaded)
        TL;DR
          * starts an agent along a single trajectory to perform curriculum
              learning, teaching the agent to do well on-distribution from
              human demonstration.
          * moves the reset point of the environment gradually backwards in
              time, once enough recent trials produce positive results,
              starting from the end of the trajectory.
        :param env: the environment to work in.
        :param traj_fname: the file from which to read the trajectory that
              the agent learns.
        :param history_len: how many recent runs should be evaluated to judge
              if the agent is ready to move its starting point backwards.
        :param reset_step: upper bound on how far the starting point can be
              moved backwards, as well randomness around the exact starting
              point.
        :param advance_limit: what fraction of recent trials must be
              successful for the starting point to be moved back.
        :param reward_thresh: what value the reward has to equal or exceed to
              count as valid for turning the reset step back.
        """
        super().__init__(env)
        self.trajectory_file = traj_fname
        self.history_len = history_len
        self.reset_step = reset_step
        self.advance_limit = advance_limit
        self.reward_thresh = reward_thresh

        # Tracks recent agent completions.
        self.recent_rewards = [0] * self.history_len
        self.traj_reward, self.last_done = 0, True
        # Actions in the trajectory
        with open(self.trajectory_file) as f:
            self.actions = json.loads(f.read())

        self.trajectory_len = len(self.actions)
        # How far from the end of the trajectory to start
        self.reset_point = 0


    def step(self, action):
        """ Normal step, saves most recent reward and done.
        """
        obs, rew, done, info = super().step(action)
        self.traj_reward += rew
        self.last_done = done
        return obs, rew, done, info

    def reset(self):
        """ If recent rollouts were successful, move the reset point back.
            Then, reset the environment and advance it on the trajectory
                for a given number of steps.
        """
        if not self.last_done:
            print("WARNING: Can only reset a backwards reset environment at the end of a trajectory!")
            print("     The current trajectory and rewards will not be counted.")
        else:
            self.recent_rewards.pop(0)
            self.recent_rewards.append(self.traj_reward >= self.reward_thresh)

            recent_avg = sum(self.recent_rewards) / len(self.recent_rewards)

            if recent_avg >= self.advance_limit:
                self.reset_point += random.randint(1, self.reset_step)
                self.recent_rewards = [0] * self.history_len
                print(f"Reset point increased to {self.reset_point}")

        self.traj_reward, self.last_done = 0, False
        obs = super().reset()


        num_steps = self.trajectory_len - self.reset_point - random.randint(1, self.reset_step)
        num_steps = max(0, num_steps)
        for i in range(num_steps):
            obs, rew, done, _ = self.step(self.actions[i])
            self.traj_reward += rew

        return obs

