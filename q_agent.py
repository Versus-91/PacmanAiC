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
from cnn import *
from constants import *
from game import GameWrapper
import random
import matplotlib
from time import sleep
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 128
TARGET_UPDATE = 10
K_FRAME = 2
SAVE_EPISODE_FREQ = 100


Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
is_reversed = (
    lambda last_action, action: "default" if REVERSED[action] -
    last_action else "reverse"
)


class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.exps = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.exps.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class LearningAgent:
    def __init__(self):
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 100000
        self.gamma = 0.99
        self.momentum = 0.95
        self.replay_size = 30000
        self.learning_rate = 0.0001
        self.steps = 0
        self.step_limit = 250000
        self.score = 0
        self.target = QNetwork().to(device)
        self.policy = QNetwork().to(device)
        # self.load_model()
        self.memory = ExperienceReplay(self.replay_size)
        self.game = GameWrapper()
        self.last_action = 0
        self.buffer = deque(maxlen=4)
        self.last_reward = -1
        self.rewards = []
        self.loop_action_counter = 0
        self.counter = 0
        self.score = 0
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

    def calculate_reward(self, done, lives, hit_wall, hit_ghost, action, prev_score):

        reward = 0
        if done:
            if lives > 0:
                print("won")
                reward = 30
            else:
                reward = -30
            return reward
        if self.score - prev_score == 10:
            reward += 10
        if self.score - prev_score == 50:
            print("power up")
            reward += 11
        if reward > 0:
            progress = self.score // 400
            reward += progress
            return reward
        if self.score - prev_score >= 200:
            return 12
        # if hit_wall:
        #     reward -= 10
        if hit_ghost:
            reward -= 20
        if REVERSED[self.last_action] == action:
            print(action, self.last_action)
            self.loop_action_counter += 1
        else:
            self.loop_action_counter = 0
        if self.loop_action_counter > 1:
            reward -= 3
            print("why the fuck")
        return reward

    def evaluate(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.counter += 1
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.cat(batch.reward)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device)
        predicted_targets = self.policy(state_batch).gather(1, action_batch)
        target_values = self.target(new_state_batch).detach().max(1)[0]
        labels = reward_batch + self.gamma * \
            (1 - dones) * target_values

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets,
                         labels.detach().unsqueeze(1)).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def act(self, state, eval=False):
        if eval:
            with torch.no_grad():
                q_values = self.policy(state)
            # Optimal action
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.counter / self.eps_decay)
        self.steps += 1
        if sample > eps_threshold:
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

    def plot_rewards(self, name="plot.png", avg=100):
        plt.figure(1)
        durations_t = torch.tensor(self.rewards, dtype=torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= avg:
            means = durations_t.unfold(0, avg, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(avg - 1), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        plt.savefig(name)

    def process_state(self, states):

        tensor = [torch.from_numpy(arr).float().to(device) for arr in states]

        # frightened_ghosts_tensor = torch.from_numpy(
        #     states[3]).float().to(device)
        channel_matrix = torch.stack(tensor, dim=0)
        channel_matrix = channel_matrix.unsqueeze(0)
        return channel_matrix

    def save_model(self):
        if self.episode % SAVE_EPISODE_FREQ == 0 and self.episode != 0 or self.steps >= self.step_limit:
            torch.save(self.policy.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"policy-model-{self.episode}-{self.steps}.pt"))
            torch.save(self.target.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"target-model-{self.episode}-{self.steps}.pt"))

    def load_model(self, name="200-44483", eval=False):
        self.steps = 44483
        self.counter = int(self.steps / 2)
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

    def map_directions(self):
        num_directions = 4

        current_direction = "up"

        direction_mapping = {
            "up": 0,
            "down": 1,
            "left": 2,
            "right": 3
        }

        direction_encoding = np.zeros(num_directions)
        direction_index = direction_mapping[current_direction]
        direction_encoding[direction_index] = 1

    def skip(self):
        if self.episode == 0:
            done = False
            while not done:
                random_action = random.choice([0, 1, 2, 3])
                obs, self.score, done, remaining_lives, invalid_move = self.game.step(
                    random_action)

    def train(self):
        self.save_model()
        obs = self.game.start()
        self.skip()
        self.score = 0
        self.episode += 1
        lives = 3
        random_action = random.choice([0, 1, 2, 3])
        obs, self.score, done, remaining_lives, invalid_move = self.game.step(
            random_action)
        state = self.process_state(obs)
        last_score = 0
        self.score = 0
        while True:
            action = self.act(state)
            action_t = action.item()
            for i in range(3):
                if not done:
                    obs, self.score, done, remaining_lives, invalid_move = self.game.step(
                        action_t)
                    if lives != remaining_lives:
                        break
                else:
                    break
            hit_ghost = False
            if lives != remaining_lives:
                hit_ghost = True
                lives -= 1
            next_state = self.process_state(obs)
            reward_ = self.calculate_reward(
                done, lives, invalid_move, hit_ghost, action_t, last_score)
            last_score = self.score
            self.memory.append(state, action,
                               torch.tensor([reward_], device=device), next_state, done)
            state = next_state
            if self.steps % 2 == 0:
                self.evaluate()
            self.last_action = action_t
            if done:
                # assert reward_sum == reward
                self.rewards.append(self.score)
                self.plot_rewards(avg=10)
                time.sleep(1)
                self.game.restart()
                torch.cuda.empty_cache()
                break

    def test(self, episodes=10):
        if self.episode < episodes:
            obs = self.game.start()
            self.episode += 1
            lives = 3
            random_action = random.choice([0, 1, 2, 3])
            obs, reward, done, remaining_lives, invalid_move = self.game.step(
                random_action)
            state = self.process_state(obs)
            while True:
                action = self.act(state, eval=True)
                action_t = action.item()
                print("action", action_t)
                for i in range(3):
                    if not done:
                        obs, reward, done, remaining_lives, invalid_move = self.game.step(
                            action_t)
                        if lives != remaining_lives:
                            break
                    else:
                        break
                if lives != remaining_lives:
                    lives -= 1
                state = self.process_state(obs)
                if done:
                    self.rewards.append(reward)
                    self.plot_rewards(name="test.png", avg=2)
                    time.sleep(1)
                    self.game.restart()
                    torch.cuda.empty_cache()
                    break
        else:
            self.game.stop()


if __name__ == '__main__':
    agent = LearningAgent()
    # agent.load_model(name="400-250134",)
    agent.rewards = []
    while True:
        agent.train()
        # agent.test()
