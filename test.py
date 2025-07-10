import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import math
from collections import deque

episodes = 100

# Neural network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ELU(),
            nn.Linear(32,16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8,action_dim),
        )

    def forward(self, x):
        return self.fc(x)

def select_action(state, epsilon=0):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return q_net(state).argmax().item()

# Initialize
env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("best_cartpole_model.pth"))
q_net.eval()

average_reward = []

# Training loop
with torch.no_grad():
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # New gym API
            state = state[0]
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state)
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += env_reward

        average_reward.append(total_reward)

        print(f"Episode {episode}, Reward: {total_reward:.0f}")

print(f"Percentage of 500 reward: {(average_reward.count(500)//episode) * 100}")
print(f"Average for 100 episodes: {np.mean(np.array(average_reward)):.2f}")

env.close()