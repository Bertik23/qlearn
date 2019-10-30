#%matplotlib inline
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pygame as pg
import pickle

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=1+2+2+1, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=3)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) # exploit

class PongEnv:
    def __init__(self, device):
        self.device = device
        self.p1Y = screen_height/2
        self.p1X = screen_width/30
        self.p2Y = screen_height/2
        self.p2X = screen_width - screen_width/30
        self.bY = screen_height/2
        self.bX = screen_width/2
        self.step = 5
        self.bDY = random.randint(-5,5)
        self.bDX = random.randint(-5,5)
        self.done = False
        self.screen = pg.display.set_mode((screen_width, screen_height))
        self.p1Score = 0
        self.p2Score = 0

    def reset(self):
        self.p1Y = screen_height/2
        self.p2Y = screen_height/2
        self.bY = screen_height/2
        self.bX = screen_width/2
        self.bXo = 0
        self.bYo = 0
        self.bDY = random.randint(-5,5)
        self.bDX = random.randint(-5,5)
        self.p1Score = 0
        self.p2Score = 0
        self.done = False

    def render(self):
        self.screen.fill((0,0,0))
        pg.draw.rect(self.screen,(0,255,0),(self.p1X-5, self.p1Y-30,10,60))
        pg.draw.rect(self.screen,(255,0,0),(self.p2X-5, self.p2Y-30,10,60))
        pg.draw.rect(self.screen,(255,255,255),(self.bX-5, self.bY-5,10,10))
        pg.display.update()

    def num_actions_available(self):
        return 3
    def take_action(self, action):
        reward = 0
        if action == 0:
            pass
        elif action == 1 and not self.p1Y + self.step > screen_height - 30:
            self.p1Y += self.step
        elif action == 2 and not self.p1Y - self.step < 30:
            self.p1Y -= self.step
        if self.p2Y > self.bY:
            self.p2Y -= self.step
        elif self.p2Y < self.bY:
            self.p2Y += self.step
        self.bXo = self.bX
        self.bYo = self.bY
        self.bX += self.bDX
        self.bY += self.bDY
        if self.p1X-5 < self.bX and self.p1X+5 > self.bX and self.p1Y-30 < self.bY and self.p1Y+30 > self.bY:
            self.bDX *= -1
            self.bDY = (self.bY - self.p1Y)//5
            reward = 10
        if self.p2X-5 < self.bX and self.p2X+5 > self.bX and self.p2Y-30 < self.bY and self.p2Y+30 > self.bY:
            self.bDX *= -1
            self.bDY = (self.bY - self.p2Y)//5
        if self.bY < 10:
            self.bDY *= -1
        if self.bY > screen_height-10:
            self.bDY *= -1
        if self.bX < 0:
            self.p2Score += 1
            self.bY = screen_height/2
            self.bX = screen_width/2
            self.bDY = random.randint(-5,5)
            self.bDX = random.randint(-5,-1)
            reward = -100
        if self.bX > screen_width:
            self.p1Score += 1
            self.bY = screen_height/2
            self.bX = screen_width/2
            self.bDY = random.randint(-5,5)
            self.bDX = random.randint(1,5)
            reward = 100
        if self.p1Score + self.p2Score >= 10:
            self.done = True
        return torch.tensor([reward], device=self.device)
    def get_state(self):
        return torch.tensor([[self.p1Y], [self.bX], [self.bY], [self.bXo], [self.bYo], [self.p2Y]]).view(-1,6)

class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=0).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

batch_size = 16
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

show_every = 250

screen_width = 600
screen_height = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = PongEnv(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN()#screen_height, screen_width).to(device)
target_net = DQN()#em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []

for episode in range(num_episodes):
    print(episode)
    em.reset()
    state = em.get_state()
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if episode % show_every == 0:
            em.render()
        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

target_net.load_state_dict(policy_net.state_dict())
with open(f"/agents/pong/agent__batch_size_{batch_size}__gamma_{gamma}__eps_start_{eps_start}__eps_end_{eps_end}__eps_decay_{eps_decay}__target_update_{target_update}__memory_size_{memory_size}__lr_{lr}__num_episodes_{num_episodes}","wb") as f:
    pickle.dump(target_net, f)

plt.show()
