from copy import deepcopy
from math import log
import os
import pygame
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import time
import numpy as np
from cnn import *
from constants import *
from game import GameWrapper
import random
import matplotlib
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize
from tensorboardX import SummaryWriter

from run_new import GameState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use("Agg")

N_ACTIONS = 4
BATCH_SIZE = 128
SAVE_EPISODE_FREQ = 100
GAMMA = 0.99
MOMENTUM = 0.95
MEMORY_SIZE = 18000

Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_START = 1.0
EPS_END = 0.1
MAX_STEP = 1000000

episodes = 1500

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

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 15, 128)
        self.fc2 = nn.Linear(128, 4)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 14 * 15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNCNN(nn.Module):
    def __init__(self):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * 7 * 7, 512)  
        self.output_layer = nn.Linear(512, 4)

    def forward(self, frame):
        x = torch.relu(self.conv1(frame))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.dense(x))
        buttons = self.output_layer(x)
        return buttons
class PacmanAgent:
    def __init__(self):
        self.steps = 0
        self.score = 0
        self.target = DQNCNN().to(device)
        self.policy = DQNCNN().to(device)
        # self.load_model()
        self.memory = ExperienceReplay(MEMORY_SIZE)
        self.game = GameWrapper()
        self.last_action = 0
        self.buffer = deque(maxlen=6)
        self.last_reward = -1
        self.rewards = []
        self.epsilon = EPS_START
        self.loop_action_counter = 0
        self.score = 0
        self.episode = 0
        self.lr = 0.00025
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.lr
        )
        self.writer = SummaryWriter('logs/dqn')
        self.prev_info =GameState()
        self.images = deque(maxlen=4)
    def get_reward(self, done, lives, hit_ghost, action, prev_score,info:GameState):
        reward = 0
        if done:
            if lives > 0:
                print("won")
                reward = 10
            else:
                reward = -10
            return reward
        progress =  int((info.collected_pellets / info.total_pellets) * 10)
        if self.score - prev_score == 10 or self.score - prev_score == 50:
            reward += 4
        if self.score >= 200:
            reward += 1
        if hit_ghost:
            reward -= 10
        reward -= 1
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
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device)

        predicted_targets = self.policy(state_batch).gather(1, action_batch)
        with torch.no_grad():
            new_state_actions = self.policy(new_state_batch).argmax(dim=1)
            target_q_values = self.target(new_state_batch).gather(1, new_state_actions.unsqueeze(1))

        # Compute the target Q-values using the Double DQN target formula
        target_values = reward_batch + GAMMA * (1 - dones) * target_q_values.squeeze()


        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets, target_values.unsqueeze(1).detach()).to(device)

        self.writer.add_scalar('loss', loss.item(), global_step=self.episode)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % 2000 == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def select_action(self, state, eval=False):
        if eval:
            with torch.no_grad():
                q_values = self.policy(state)
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        rand = random.random()
        self.steps += 1
        if rand > self.epsilon:
            with torch.no_grad():
                q_values = self.policy(state)
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

        tensors = [arr.to(device) for arr in states]
        # frightened_ghosts_tensor = torch.from_numpy(
        #     states[3]).float().to(device)
        channel_matrix = torch.cat(tensors, dim=1)
        #channel_matrix = channel_matrix.unsqueeze(0)
        return channel_matrix

    def save_model(self):
        if self.episode % SAVE_EPISODE_FREQ == 0 and self.episode != 0:
            torch.save(self.policy.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"policy-model-{self.episode}-{self.steps}.pt"))
            torch.save(self.target.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"target-model-{self.episode}-{self.steps}.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"optimizer-{self.episode}-{self.steps}.pt"))

    def load_model(self, name, eval=False):
        name_parts = name.split("-")
        self.episode = int(name_parts[0])
        self.steps = int(name_parts[1])
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
            path = os.path.join(
                os.getcwd() + "\\results", f"optimizer-{name}.pt")
            self.optimizer.load_state_dict(torch.load(path))
            self.target.train()
            self.policy.train()
    def pacman_pos(self,state):
        index = np.where(state != 0)
        if len(index[0]) != 0:
            x = index[0][0]
            y = index[1][0]
            return (x,y)
        return None
    def plot(self):
            if len(self.images) == 4:
                images = deepcopy(self.images)
                fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                for i in range(2):
                    for j in range(2):
                        # Get the next image tensor from the deque
                        image_tensor = images.popleft()
                        image_tensor = image_tensor.squeeze()
                        # Convert the tensor to a NumPy array
                        image_array = image_tensor.numpy()

                        # Convert the array to grayscale if necessary
                        # (e.g., if the tensor is in the shape (C, H, W))
                        # image_array = np.mean(image_array, axis=0)

                        # Plot the image
                        axs[i, j].imshow(image_array)
                        axs[i, j].axis('off')

                    # Adjust the spacing between subplots
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                # Display the plo
                plt.pause(0.001)
                plt.savefig("frames.png")
    def processs_image(self,screen):
        screen = np.transpose(screen, (1, 0, 2))
        screen_tensor = to_tensor(screen).unsqueeze(0)
        resized_tensor = F.interpolate(screen_tensor, size=(92, 84), mode='area')
        grayscale_tensor = resized_tensor.mean(dim=1, keepdim=True)
        crop_pixels_top = 4
        crop_pixels_bottom = 4
        height = grayscale_tensor.size(2)
        cropped_tensor = grayscale_tensor[:, :, crop_pixels_top:height - crop_pixels_bottom, :]
        normalized_tensor = cropped_tensor / 255.0
        # image_array = normalized_tensor.squeeze().numpy()
        # plt.imshow(image_array)
        # plt.show()
        return normalized_tensor
    def train(self):
        while self.steps <= MAX_STEP:
            self.save_model()
            obs = self.game.start()
            self.episode += 1
            random_action = random.choice([0, 1, 2, 3])
            obs, self.score, done, info = self.game.step(
                random_action)
            last_score = 0
            lives = 3
            for i in range(6):
                obs, self.score, done, info = self.game.step(random_action)      
                self.images.append(self.processs_image(info.image))
            state = self.process_state(self.images)
            while True:
                action = self.select_action(state)
                action_t = action.item()
                for i in range(4):
                    obs, self.score, done, info = self.game.step(
                            action_t)
                    if lives != info.lives or done:
                        break
                hit_ghost = False
                if lives != info.lives:
                    self.plot()
                    hit_ghost = True
                    lives -= 1
                self.images.append(self.processs_image(info.image))
                reward_ = self.get_reward(done, lives, hit_ghost, action_t, last_score, info)
                self.prev_info = info
                last_score = self.score
                next_state = self.process_state(self.images)
                self.memory.append(state, action,torch.tensor([reward_], device=device), next_state, done)
                state = next_state
                if self.steps % 2 == 0:
                    self.optimize_model()
                if not info.invalid_move:
                    self.last_action = action_t
                if done:
                    self.epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END)* self.steps / MAX_STEP)
                    self.writer.add_scalar('episode reward', self.score, global_step=self.episode)
                    self.log()
                    # assert reward_sum == reward
                    self.rewards.append(self.score)
                    self.game.restart()
                    self.plot_rewards(avg=50,name="double_dqn.png")
                    torch.cuda.empty_cache()
                    break
        else:
            self.save_model()
            exit()
    def log(self):
        # current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            "epsilon",
            round(self.epsilon, 3),
            "reward",
            self.score,
            "learning rate",
            self.lr,
            "episode",
            self.episode,
            "steps",
            self.steps,
        )
    def test(self, episodes=10):
        if self.episode < episodes:
            obs = self.game.start()
            self.episode += 1
            random_action = random.choice([0, 1, 2, 3])
            obs, self.score, done, info = self.game.step(
                random_action)
            last_score = 0
            lives = 3
            for i in range(6):
                obs, self.score, done, info = self.game.step(random_action)      
                self.images.append(self.processs_image(info.image))
            state = self.process_state(self.images)
            while True:
                action = self.select_action(state,True)
                action_t = action.item()
                for i in range(4):
                    obs, self.score, done, info = self.game.step(
                            action_t)
                    if lives != info.lives or done:
                        break
                if lives != info.lives:
                    lives -= 1
                self.images.append(self.processs_image(info.image))
                self.prev_info = info
                self.images.append(self.processs_image(info.image))
                state = self.process_state(self.images)
                if done:
                    self.rewards.append(self.score)
                    self.plot_rewards(name="test.png", avg=2)
                    time.sleep(1)
                    self.game.restart()
                    torch.cuda.empty_cache()
                    break
        else:
            self.game.stop()


if __name__ == '__main__':
    agent = PacmanAgent()
    #agent.load_model(name="1500-746581", eval=True)
    agent.episode = 0
    agent.rewards = []
    while True:
        agent.train()
        #agent.test()
