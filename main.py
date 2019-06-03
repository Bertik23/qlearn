import gym
import numpy as np

env = gym.make("MountainCar-v0")
#env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 2000 #th episode

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    disctere_state = (state  - env.observation_space.low) / discrete_os_win_size
    return tuple(disctere_state.astype(np.int))

for episode in range(EPISODES):
    disctere_state = get_discrete_state(env.reset())

    done = False
    if episode % SHOW_EVERY == 0:
        #env.render()
        print(f"HI I am episode {episode}")
    while not done:
        if episode % SHOW_EVERY == 0:
            env.render()
            #print(f"HI I am episode {episode}")
        action = np.argmax(q_table[disctere_state])
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[disctere_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[disctere_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[disctere_state + (action, )] = 0
            print(f"Made it! I got there on episode {episode}")
        disctere_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
