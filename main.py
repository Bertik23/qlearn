#THIS IS WHAT I GOT SO FAR
import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("MountainCar-v0")
#env.reset()

LEARNING_RATE = 1
DISCOUNT = 1
EPISODES = 2000

SHOW_EVERY = 200 #th episode

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {"ep": [], "avg": [], "min": [], "max": []}

def get_discrete_state(state):
    disctere_state = (state  - env.observation_space.low) / discrete_os_win_size
    return tuple(disctere_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward = 0
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
        episode_reward += reward
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
    ep_rewards.append(episode_reward)
    if episode % SHOW_EVERY == 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards["max"].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Ep: {episode}, Avg: {average_reward}, Min: {min(ep_rewards[-SHOW_EVERY:])}, Max: {max(ep_rewards[-SHOW_EVERY:])}")
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()

plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label = "avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label = "min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label = "max")
plt.legend(loc = 4)
plt.show()
print(q_table)
