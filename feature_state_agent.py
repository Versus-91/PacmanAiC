import os
import numpy as np
import torch.optim as optim
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import time
from cnn import Conv2dNetwork
from game import GameWrapper
import random
import matplotlib
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn as nn

from run import GameState

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 128
SAVE_EPISODE_FREQ = 100
GAMMA = 0.99
sync_every = 100
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500000
MAX_STEPS = 600000

class PacmanNet(nn.Module):
    def __init__(self):
        super(PacmanNet, self).__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ExperienceReplay:
    def __init__(self, capacity) -> None:
        self.exps = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.exps.append(Experience(state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)


class PacmanAgent:
    def __init__(self):
        self.steps = 0
        self.score = 0
        self.target = PacmanNet().to(device)
        self.policy = PacmanNet().to(device)
        self.memory = ExperienceReplay(20000)
        self.game = GameWrapper()
        self.lr = 0.001
        self.writer = SummaryWriter('logs/dqn')
        self.current_direction = 0
        self.buffer = deque(maxlen=4)
        self.last_reward = -1
        self.rewards = []
        self.loop_action_counter = 0
        self.score = 0
        self.episode = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        self.losses = []
    def calculate_reward(
        self, done, lives, hit_ghost, action, prev_score, info: GameState
    ):
        reward = 0
        if done:
            if lives > 0:
                print("won")
                reward = 50
            else:
                reward = -100
            return reward
        if self.score - prev_score == 10:
            reward += 12
        if self.score - prev_score == 50:
            print("power up")
            reward += 2
        if self.score - prev_score >= 200:
            reward += 20 
        if hit_ghost:
            reward -= 50
        index = np.where(info.frame == 5)
        if len(index[0]) != 0:
            x = index[0][0]
            y = index[1][0]
            try:
                upper_cell = info.frame[x + 1][y]
                lower_cell = info.frame[x - 1][y]
            except IndexError:
                upper_cell = 0
                lower_cell = 0
                print("x",index[0][0],"y",index[1][0])
            try:
                right_cell = info.frame[x][y + 1]
                left_cell = info.frame[x][y - 1]
            except IndexError:
                right_cell = 0
                left_cell = 0
                print("x",index[0][0],"y",index[1][0])

            if action == 0:
                if upper_cell == 1:
                    reward -= 10
            elif action == 1:
                if lower_cell == 1:
                    reward -= 10
            elif action == 2:
                if left_cell == 1:
                    reward -= 10
            elif action == 3:
                if right_cell == 1:
                    reward -= 10
        reward = round(reward, 2)
        if REVERSED[self.current_direction] == action:
            reward -= 6
        reward -= 5
        #print(reward)
        return reward

    def write_matrix(self, matrix):
        with open("outfile.txt", "wb") as f:
            for line in matrix:
                np.savetxt(f, line, fmt="%.2f")

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        new_state_batch = torch.stack(batch.new_state)
        reward_batch = torch.cat(batch.reward)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device)
        predicted_targets = self.policy(state_batch).gather(1, action_batch)
        target_values = self.target(new_state_batch).detach().max(1)[0]
        labels = reward_batch + GAMMA * (1 - dones) * target_values
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets, labels.detach().unsqueeze(1)).to(device)
        self.writer.add_scalar('loss', loss.item(), global_step=self.steps)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.steps % sync_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def act(self, state, eval=False):
        if eval:
            with torch.no_grad():
                q_values = self.policy(state)
            return torch.argmax(q_values)
        rand = random.random()
        epsilon = max(
            EPS_END, EPS_START - (EPS_START - EPS_END) * (self.steps) / EPS_DECAY
        )
        self.steps += 1
        if rand > epsilon:
            with torch.no_grad():
                outputs = self.policy(state)
            res = torch.argmax(outputs)
            print("predicted action", res.item())
            return torch.argmax(outputs)

        else:
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.current_direction]:
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def plot_rewards(self,items, name="plot.png",label="rewards", avg=100):
        plt.figure(1)
        rewards = torch.tensor(items, dtype=torch.float)
        plt.xlabel("Episode")
        plt.ylabel(label)
        plt.plot(rewards.numpy())
        if len(rewards) >= avg:
            means = rewards.unfold(0, avg, 1).mean(1).view(-1)
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

    def save_model(self, force=False):
        if (self.episode % SAVE_EPISODE_FREQ == 0 and self.episode != 0) or force:
            torch.save(
                self.policy.state_dict(),
                os.path.join(
                    os.getcwd() + "\\results",
                    f"policy-model-{self.episode}-{self.steps}.pt",
                ),
            )
            torch.save(
                self.target.state_dict(),
                os.path.join(
                    os.getcwd() + "\\results",
                    f"target-model-{self.episode}-{self.steps}.pt",
                ),
            )

    def load_model(self, name, eval=False):
        path = os.path.join(os.getcwd() + "\\results", f"target-model-{name}.pt")
        self.target.load_state_dict(torch.load(path))
        path = os.path.join(os.getcwd() + "\\results", f"policy-model-{name}.pt")
        self.policy.load_state_dict(torch.load(path))
        if eval:
            self.target.eval()
            self.policy.eval()
        else:
            name_parts = name.split("-")
            self.episode = int(name_parts[0])
            self.steps = int(name_parts[1])
            scheduler_steps = round(self.steps // 100000)
            for i in range(scheduler_steps):
                self.scheduler.step()
            self.target.train()
            self.policy.train()
    def map_direction(self,dir):
        if dir == 1:
            action = 0
        elif dir == -1:
            action = 1
        elif dir == 2:
            action = 2
        elif dir == -2:
            action = 3
        else:
            action = random.randrange(N_ACTIONS)
        return action
    def train(self):
        if self.steps >= MAX_STEPS:
            self.save_model(force=True)
            exit()
        self.save_model()
        obs = self.game.start()
        self.episode += 1
        random_action = random.choice([0, 1, 2, 3])
        obs, self.score, done, info = self.game.step(random_action)
        #state = self.process_state(obs)
        state = torch.tensor(info.frame).float().to(device)
        last_score = 0
        lives = 3
        reward_total = 0
        while True:
            action = self.act(state)
            action_t = action.item()
            for i in range(2):
                if not done:
                    obs, self.score, done, info = self.game.step(action_t)
                    if lives != info.lives or self.score - last_score != 0:
                        break
                else:
                    break
            self.buffer.append(info.frame)
            hit_ghost = False
            if lives != info.lives:
                hit_ghost = True
                lives -= 1
                if not done:
                    for i in range(3):
                        _, _, _, _ = self.game.step(action_t)
            next_state = torch.tensor(info.frame).float().to(device)
            reward_ = self.calculate_reward(
                done, lives, hit_ghost, action_t, last_score, info
            )
            reward_total += reward_
            last_score = self.score
            action_tensor = torch.tensor([[action_t]], device=device, dtype=torch.long)
            self.memory.append(
                state,
                action_tensor,
                torch.tensor([reward_], device=device),
                next_state,
                done,
            )
            state = next_state
            self.learn()
            self.current_direction = self.map_direction(info.direction)
            if self.steps % 100000 == 0:
                self.scheduler.step()
            if done:
                self.log()
                self.rewards.append(self.score)
                self.plot_rewards(items= self.rewards, avg=50)
                #self.plot_rewards(items = self.losses,label="losses",name="losses.png", avg=50)
                time.sleep(1)
                self.game.restart()
                torch.cuda.empty_cache()
                break

    def log(self):
        current_lr = self.optimizer.param_groups[0]["lr"]
        epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (self.steps) / EPS_DECAY,
        )
        print(
            "epsilon",
            round(epsilon, 3),
            "reward",
            self.score,
            "learning rate",
            current_lr,
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
            obs, self.score, done, info = self.game.step(random_action)
            state = torch.tensor(info.frame).float().to(device)
            while True:
                action = self.act(state, eval=True)
                action_t = action.item()
                for i in range(3):
                    if not done:
                        obs, reward, done, info = self.game.step(action_t)
                    else:
                        break
                state = torch.tensor(info.frame).float().to(device)
                if done:
                    self.rewards.append(reward)
                    self.plot_rewards(name="test.png",items=self.rewards, avg=2)
                    #self.writer.add_scalar('episode reward', reward, global_step=self.episode)
                    time.sleep(1)
                    self.game.restart()
                    torch.cuda.empty_cache()
                    break
        else:
            exit()


if __name__ == "__main__":
    agent = PacmanAgent()
    agent.load_model(name="400-298452", eval=False)
    agent.rewards = []
    while True:
        agent.train()
        #agent.test()
