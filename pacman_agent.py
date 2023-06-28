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

from run import GameState

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4
BATCH_SIZE = 128
SAVE_EPISODE_FREQ = 100
GAMMA = 0.99
MOMENTUM = 0.95
sync_every = 100
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 500000
MAX_STEPS = 600000


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
        self.target = Conv2dNetwork().to(device)
        self.policy = Conv2dNetwork().to(device)
        self.memory = ExperienceReplay(20000)
        self.game = GameWrapper()
        self.lr = 0.001
        self.writer = SummaryWriter('logs/dqn')
        self.last_action = 0
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
        self, done, lives, hit_ghost, action, prev_score, info: GameState, state
    ):
        reward = 0
        time_penalty = -0.01
        movement_penalty = -0.1
        progress = round((info.collected_pellets / info.total_pellets) * 10)
        if done:
            if lives > 0:
                print("won")
                reward = 50
            else:
                reward = -50
            return reward
        if self.score - prev_score == 10:
            reward += 1
        if self.score - prev_score == 50:
            print("power up")
            reward += 5
        if reward > 0:
            reward += progress
        if self.score - prev_score >= 200:
            reward += 20 * ((self.score - prev_score) / 200)
        if info.invalid_move:
            reward -= 1
        if hit_ghost:
            reward -= 30
        reward += time_penalty
        reward += movement_penalty
        index = np.where(state == 5)
        if len(index[0]) != 0:
            x = index[0][0]
            y = index[1][0]
            try:
                n1 = state[x + 1][y]
                n2 = state[x - 1][y]
            except IndexError:
                n1 = 0
                n2 = 0
                print("x",index[0][0],"y",index[1][0])
            try:
                n3 = state[x][y + 1]
                n4 = state[x][y - 1]
            except IndexError:
                n3 = 0
                n4 = 0
                print("x",index[0][0],"y",index[1][0])
            if -6 in (n1, n2, n3, n4):
                reward -= 30
            elif 3 in (n1, n2, n3, n4):
                reward += 1 + progress
            elif 4 in (n1, n2, n3, n4):
                reward += 3 + progress
        reward = round(reward, 2)
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
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        new_state_batch = torch.cat(batch.new_state)
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
            vals = q_values.max(1)[1]
            return vals.view(1, 1)
        rand = random.random()
        epsilon = max(
            EPS_END, EPS_START - (EPS_START - EPS_END) * (self.steps) / EPS_DECAY
        )
        self.steps += 1
        if rand > epsilon:
            with torch.no_grad():
                outputs = self.policy(state)
            return outputs.max(1)[1].view(1, 1)
        else:
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.last_action]:
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
            self.target.train()
            self.policy.train()

    def train(self):
        if self.steps >= MAX_STEPS:
            self.save_model(force=True)
            exit()
        self.save_model()
        obs = self.game.start()
        self.episode += 1
        random_action = random.choice([0, 1, 2, 3])
        # obs, self.score, done, info = self.game.step(random_action)
        # state = self.process_state(obs)
        # state = torch.tensor(obs).float().to(device)
        for i in range(6):
            obs, self.score, done, info = self.game.step(random_action)
            self.buffer.append(obs)
        state = self.process_state(self.buffer)
        last_score = 0
        lives = 3
        reward_total = 0
        while True:
            action = self.act(state)
            action_t = action.item()
            for i in range(3):
                if not done:
                    obs, self.score, done, info = self.game.step(action_t)
                    if lives != info.lives or self.score - last_score != 0:
                        break
                else:
                    break
            self.buffer.append(obs)
            hit_ghost = False
            if lives != info.lives:
                # self.write_matrix(self.buffer)
                hit_ghost = True
                lives -= 1
                if not done:
                    for i in range(3):
                        _, _, _, _ = self.game.step(action_t)
            # next_state = torch.tensor(obs).float().to(device)
            next_state = self.process_state(self.buffer)

            reward_ = self.calculate_reward(
                done, lives, hit_ghost, action_t, last_score, info, obs
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
            self.last_action = action_t
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
            for i in range(6):
                obs, self.score, done, info = self.game.step(random_action)
                self.buffer.append(obs)
            state = self.process_state(self.buffer)
            while True:
                action = self.act(state, eval=True)
                action_t = action.item()
                for i in range(3):
                    if not done:
                        obs, reward, done, info = self.game.step(action_t)
                    else:
                        break
                self.buffer.append(obs)
                state = self.process_state(self.buffer)
                if done:
                    self.rewards.append(reward)
                    #self.plot_rewards(name="test.png",items=self.rewards, avg=2)
                    self.writer.add_scalar('episode reward', reward, global_step=self.episode)
                    time.sleep(1)
                    self.game.restart()
                    torch.cuda.empty_cache()
                    break
        else:
            exit()


if __name__ == "__main__":
    agent = PacmanAgent()
    #agent.load_model(name="1200-511012", eval=True)
    agent.rewards = []
    while True:
        agent.train()
        #agent.test()
