import gym
import matplotlib
import time
import numpy as np

# getting environment
env = gym.make("MountainCar-v0")
env.reset()

# lowest and highest possible values
print(env.observation_space.high)
print(env.observation_space.low)

# step size
print(env.action_space.n)

  
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

done = False

while not done:
    # step through environment
    
    # action = 0 : push car left
    # action = 1 : do nothing
    # action = 2 : push car right
    action = 2
    
    # new_state = position, velocit
    new_state, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.1)

env.close()
