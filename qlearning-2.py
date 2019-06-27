import gym
import matplotlib
import time
import numpy as np

# getting environment
env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

# some kind of weihgt, measuring how important is value
DISCOUNT = 0.95
EPISODES = 25000
SHOW = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# apply random action, higher epsilon more likely do random action
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

# amount of decay value for each episode
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# initializing Q-table, table size: 20, 20, 3 - all possible combinations for all actions
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    """
    converts given continuous state to discrete state
    :param state: current state
    :return: discrete state
    """

    discrete_state = (state - env.observation_space.low) / discrete_os_window_size
    return tuple(discrete_state.astype(np.int))


# print("Discrete state: {}".format(discrete_state))
# print("q_table[discrete_state] = {}".format(q_table[discrete_state]))
#
# # action value
# action = np.argmax(q_table[discrete_state])
# print("np.argmax(q_table[discrete_state]) = {}".format(action))

for episode in range(EPISODES):

    if episode % SHOW == 0:
        print("{}/{}".format(episode, EPISODES))
        render = True
    else:
        render = False

    # env.reset gives us initial state and
    # we convert it to discrete
    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        # step through environment

        # action = 0 : push car left
        # action = 1 : do nothing
        # action = 2 : push car right

        if np.random.random() > epsilon:
            # choose action from q_table
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # new_state = position, velocity
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
            time.sleep(0.01)

        # if game doesn't end, update q table
        if not done:
            # estimate of optimal future value
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            # calculating new_q value according to the formula
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # print("current_q: {}".format(current_q))
            # print("new_q: {}".format(new_q))

            # updating q value
            q_table[discrete_state + (action,)] = new_q

        # if goal reached, set punishment 0
        elif new_state[0] >= env.goal_position:
            # print("Goal reached at {}".format(e))
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
env.close()
