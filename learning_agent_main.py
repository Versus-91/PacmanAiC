from math import log
import math
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import time
import cv2
import numpy as np
from cnn import NeuralNetwork
from constants import *
from game import GameWrapper
import random
import matplotlib
from time import sleep

from run import GameState
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 128
TARGET_UPDATE = 60  # here
K_FRAME = 2
SAVE_EPISODE_FREQ = 100
def optimization(it, r): return it % K_FRAME == 0 and r


Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
is_reversed = (
    lambda last_action, action: "default" if REVERSED[action] -
    last_action else "reverse"
)


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class LearningAgent:
    def __init__(self):
        # self.eps_start = 0.9
        # self.eps_end = 0.05
        self.eps_start = 1
        self.eps_end = 0.1
        self.eps_decay = 400000
        self.gamma = 0.99
        self.momentum = 0.95
        self.replay_size = 20000
        self.learning_rate = 0.001
        self.steps = 0
        self.target = NeuralNetwork().to(device)
        self.policy = NeuralNetwork().to(device)
        # self.load_model()
        self.memory = ExperienceReplay(self.replay_size)
        self.game = GameWrapper()
        self.last_action = 0
        self.rewards = []
        self.loss = []
        self.episode = 0
        self.optimizer = optim.SGD(
            self.policy.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True
        )

    def calculate_distance(pos1, pos2):
        # pos1 and pos2 are tuples representing positions (x, y)
        x1, y1 = pos1
        x2, y2 = pos2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def calculate_reward(self, done, lives, eat_pellet, eat_powerup, hit_wall, hit_ghost, ate_ghost, info: GameState):
        # Initialize reward
        reward = 0

        # Check if the game is won or lost
        if done:
            if lives > 0:
                reward = 100  # Game won
            else:
                reward = -100  # Game lost
            return reward

        if eat_pellet:
            # Pacman ate a pellet
            reward += 10 + (info.collected_pellets / info.total_pellets) * 15
        if eat_powerup:
            reward += 30 + (info.collected_pellets / info.total_pellets) * 15

        # Encourage Pacman to move towards the nearest pellet
        # reward -= distance_to_pellet

        # Penalize Pacman for hitting walls or ghosts
        if hit_wall:
            reward -= 5  # Pacman hit a wall
        if hit_ghost:
            reward -= 50  # Pacman hit a ghost
        if ate_ghost:
            reward += 30
        return reward

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.cat(batch.reward)
        indices = random.sample(range(len(experiences)), k=BATCH_SIZE)
        def extract(list_): return [list_[i] for i in indices]
        done_array = [s for s in batch.done]
        dones = torch.from_numpy(
            np.vstack(extract(done_array)).astype(np.uint8)).to(device)
        predicted_targets = self.policy(state_batch).gather(1, action_batch)
        target_values = self.target(new_state_batch).detach().max(1)[0]
        labels = reward_batch + self.gamma * \
            (1 - dones.squeeze(1)) * target_values

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets,
                         labels.detach().unsqueeze(1)).to(device)
        self.loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())
        # # Softmax update
        # for target_param, local_param in zip(target_DQN.parameters(), policy_DQN.parameters()):
        #     target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)

    def act(self, state, eval=False):
        sample = random.random()
        if eval:
            epsilon = 0.05
            if sample > epsilon:
                with torch.no_grad():
                    q_values = self.policy(state)
                # Optimal action
                vals = q_values.max(1)[1]
                return vals.view(1, 1)
            else:
                action = random.randrange(N_ACTIONS)
                while action == REVERSED[self.last_action]:
                    action = random.randrange(N_ACTIONS)
                return torch.tensor([[action]], device=device, dtype=torch.long)
        epsilon = max(self.eps_end, self.eps_start -
                      (self.eps_start - self.eps_end) * self.steps / self.eps_decay)
        self.steps += 1
        # display.data.q_values.append(q_values.max(1)[0].item())
        if sample > epsilon:
            with torch.no_grad():
                q_values = self.policy(state)
            # Optimal action
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        else:
            # Random action
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.last_action]:
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def plot_rewards(self, name="progress", avg=50, items=[]):
        plt.figure(1)
        durations_t = torch.tensor(items, dtype=torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= avg:
            means = durations_t.unfold(0, avg, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(avg-1), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
        plt.savefig(name+'.png')

    def process_state(self, states):
        walls_tensor = torch.from_numpy(states[0]).float().to(device)
        pacman_tensor = torch.from_numpy(states[1]).float().to(device)
        pellets_tensor = torch.from_numpy(states[2]).float().to(device)
        ghosts_tensor = torch.from_numpy(states[3]).float().to(device)
        # ghosts_tensor = torch.from_numpy(states[4]).float().to(device)
        # frightened_ghosts_tensor = torch.from_numpy(
        #     states[5]).float().to(device)
        channel_matrix = torch.stack([walls_tensor, pacman_tensor, pellets_tensor,
                                     ghosts_tensor], dim=0)
        channel_matrix = channel_matrix.unsqueeze(0)
        return channel_matrix

    def save_model(self):
        if self.episode % SAVE_EPISODE_FREQ == 0 and self.episode != 0:
            torch.save(self.policy.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"policy-model-{self.episode}-{self.steps}.pt"))
            torch.save(self.target.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"target-model-{self.episode}-{self.steps}.pt"))

    def load_model(self, name="200-44483", eval=True):
        self.steps = 44483
        path = os.path.join(
            os.getcwd() + "\\results", f"target-model-{name}.pt")
        self.target.load_state_dict(torch.load(path))
        path = os.path.join(
            os.getcwd() + "\\results", f"policy-model-{name}.pt")
        self.policy.load_state_dict(torch.load(path))
        if eval:
            self.target.eval()
            self.policy.eval()
        else:
            self.target.train()
            self.policy.train()

    def train(self):
        self.save_model()
        obs = self.game.start()
        action_interval = 0.03
        start_time = time.time()
        self.episode += 1
        lives = 3
        action = random.choice([0, 1, 2, 3])
        obs, reward, done, info, = self.game.step(action)
        # obs = obs[0].flatten().astype(dtype=np.float32)
        # state = torch.from_numpy(obs).unsqueeze(0).to(device)
        state = self.process_state(obs)
        reward_sum = 0
        last_score = 0
        self.last_action = action
        last_state = None
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= action_interval:
                action = self.act(state)
                action_t = action.item()
                obs, reward, done, info = self.game.step(
                    action_t)
                hit_ghost = False
                if lives != info.lives:
                    hit_ghost = True
                    lives -= 1
                next_state = self.process_state(obs)
                reward_ = self.calculate_reward(
                    done, lives, reward - last_score == 10, reward - last_score == 50, info.invalid_move, hit_ghost, reward - last_score >= 200, info)
                if last_score < reward:
                    reward_sum += reward - last_score
                last_score = reward
                action_tensor = torch.tensor(
                    [[action_t]], device=device, dtype=torch.long)
                self.memory.append(state, action_tensor,
                                   torch.tensor([reward_], device=device), next_state, done)
                state = next_state
                last_state = next_state
                self.last_action = action_t
                if self.steps % 2 == 0:
                    self.optimize_model()
                start_time = time.time()
                if self.steps % 100000 == 0:
                    if self.steps // 100000 <= 3:
                        lr = self.optimizer.param_groups[0]['lr']
                        self.optimizer.param_groups[0]['lr'] = lr * 0.8
            elif elapsed_time < action_interval:
                obs, reward, done, info = self.game.step(
                    self.last_action)
                if done:
                    reward_ = -100
                    next_state = self.process_state(obs)
                    self.memory.append(last_state, action_tensor,
                                       torch.tensor([reward_], device=device), next_state, done)
            if done:
                current_lr = self.optimizer.param_groups[0]["lr"]
                epsilon = max(self.eps_end, self.eps_start -
                              (self.eps_start - self.eps_end) * self.steps / self.eps_decay)
                print("epsilon", round(epsilon, 3), "reward", reward_sum, "learning_rate",
                      current_lr, "steps", self.steps, "episode", self.episode)
                # assert reward_sum == reward
                self.rewards.append(reward_sum)
                self.plot_rewards(avg=50, items=self.rewards,
                                  name="rewards")
                time.sleep(1)
                self.game.restart()
                reward_sum = 0
                torch.cuda.empty_cache()
                break

    def test(self, model=""):
        self.load_model(name=model)
        obs = self.game.start()
        action_interval = 0.03
        start_time = time.time()
        self.episode += 1
        obs, reward, done, info = self.game.step(2)
        state = self.process_state(obs)
        reward_sum = 0
        self.last_action = 2

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= action_interval:
                action = self.act(state, eval=True)
                action_t = action.item()
                obs, reward, done, info = self.game.step(
                    action_t)
                state = self.process_state(obs)
                self.last_action = action_t
                start_time = time.time()
                reward_sum = reward
            elif elapsed_time < action_interval:
                obs, reward, done, info = self.game.step(
                    self.last_action)
                reward_sum = reward
            if done:
                # assert reward_sum == reward
                self.rewards.append(reward_sum)
                self.plot_rewards(avg=10, name="test", items=self.rewards)
                time.sleep(1)
                self.game.restart()
                reward_sum = 0
                torch.cuda.empty_cache()
                break


if __name__ == '__main__':
    agent = LearningAgent()
    while True:
        agent.train()
        # agent.test(model="300-107646")
