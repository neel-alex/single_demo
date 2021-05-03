import time
import json

import gym

env = gym.make("SpaceInvadersDeterministic-v4")

obs = env.reset()
done = False
r = 0

with open("action_seq.json") as f:
    action_seq = json.loads(f.read())

print(f"Loaded {len(action_seq)} actions!")

while not done:
    time.sleep(0.01)
    env.render()
    if action_seq == []:
        action = 0
    else:
        action = action_seq[0]    
    obs, rew, done, info = env.step(action)
    action_seq = action_seq[1:]
    r += rew

print(f"Trajectory achieved {r} reward.")
