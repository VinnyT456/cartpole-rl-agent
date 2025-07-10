import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import math
from collections import deque
from model import QNetwork
from utils import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters
gamma = 0.99
lr = 1e-3
epsilon = 1.0
epsilon_decay = 0.990
epsilon_min = 0.01
batch_size = 128
target_update = 10
memory_size = 10000
episodes = 1000

average_reward = []

# Replay buffer
memory = []
priorities = []
pos = 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return q_net(state).argmax().item()
    
def sample_per(batch_size, alpha=0.6, beta=0.4):
    current_len = len(memory)
    prios = np.array(priorities[:current_len])

    probs = prios ** alpha
    probs /= probs.sum()

    indices = np.random.choice(current_len, batch_size, p=probs)
    samples = [memory[i] for i in indices]

    weights = ((1 / current_len) * 1 / (probs[indices])) ** beta
    weights /= weights.max()

    weights = torch.tensor(weights, dtype=torch.float32)

    return samples, indices, weights

def train_model():
    #Start training the neural network whenever the memory gets past the batch_size
    if len(memory) < batch_size:
        return 
    #Randomly choose a sample size of (batch_size) from the memory for training
    batch, indices, weights = sample_per(batch_size)

    #Split it into state, action, reward, next state, and done
    s, a, r, s_, d = zip(*batch)

    #Turn it all into tensors and add an extra dimension if necessary
    s = torch.from_numpy(np.array(s)).float().to(device)
    a = torch.LongTensor(a).unsqueeze(1).to(device)
    r = torch.FloatTensor(r).unsqueeze(1).to(device)
    s_ = torch.from_numpy(np.array(s_)).float().to(device)
    d = torch.FloatTensor(d).unsqueeze(1).to(device)

    #Grab the q values in each row based on the index given by the actions
    q_pred = q_net(s).gather(1, a)

    #Get the action based on the max q value from each row using the next state
    online_next_actions = q_net(s_).argmax(dim=1, keepdim=True)

    #Grab the q values in each row based on the actions and detach it 
    next_q = target_net(s_).gather(1, online_next_actions).detach() 

    #Find the new q value with the formula
    q_target = r + gamma * next_q * (1 - d)

    #Compute loss, zero gradience, back propagation, step the optimizer
    loss = (weights * (q_pred - q_target).pow(2)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    td_errors = (q_pred - q_target).abs().detach()
    new_priorities = td_errors + 1e-5

    for idx, prio in zip(indices, new_priorities):
        priorities[idx] = prio.item()
    
# Initialize
env = gym.make("CartPole-v1")

env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=lr)

q_net.train()
target_net.eval()

max_reward = -float('inf')

# Training loop
for episode in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):  # New gym API
        state = state[0]
    total_reward = 0
    done = False

    step = 0
    episode_buffer = []
    
    while not done:
        action = select_action(state, epsilon)
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        shaped_reward = env_reward + 0.1 * custom_reward(state,action)
        
        step += 1

        if (step >= 10):
            episode_buffer.append((state, action, shaped_reward, next_state, done))

        state = next_state
        total_reward += env_reward
        train_model()

    max_priority = max(priorities, default=1.0) if (total_reward != 500) else 10.0
    for transition in episode_buffer:
        if (len(memory) < memory_size):
            memory.append(transition)
            priorities.append(max_priority)
        else:
            memory[pos] = transition
            priorities[pos] = max_priority
            pos = (pos + 1) % memory_size

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())

    average_reward.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    if (total_reward > max_reward):
        max_reward = total_reward
        torch.save(q_net.state_dict(), "best_cartpole_model.pth")

print(f"Percentage of 500 reward: {(average_reward.count(500)/episode) * 100}")
print(f"Average for last 100 episodes: {np.mean(np.array(average_reward[-100:])):.3f}")

env.close()