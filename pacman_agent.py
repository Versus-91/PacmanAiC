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
from constants import *
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
sync_every = 100
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
EPS_START = 0.95
EPS_END = 0.05
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
        self.memory = ExperienceReplay(13000)
        self.game = GameWrapper()
        self.lr = 0.0003
        self.writer = SummaryWriter('logs/dqn')
        self.current_direction = 0
        self.buffer = deque(maxlen=4)
        self.last_reward = -1
        self.rewards = []
        self.loop_action_counter = 0
        self.score = 0
        self.episode = 0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        self.losses = []
        self.prev_info=[]
        self.last_action = 0
    def get_reward(self, done, lives, hit_ghost, action, prev_score,info:GameState):
        reward = 0
        if done:
            if lives > 0:
                print("won")
                reward = 30
            else:
                reward = -30
            return reward
        
        progress =  int((info.collected_pellets / info.total_pellets) * 7)
        if self.score - prev_score == 10:
            reward += 10
        if self.score - prev_score == 50:
            reward += 13
        if reward > 0:
            reward += progress
            return reward
        if self.score - prev_score >= 200:
            return 16
        invalid_in_maze= self.get_neighbors(info,action)        
        if hit_ghost:
            reward -= 20
        # if (info.ghost_distance >=1 and info.ghost_distance < 8):
        #     if  self.prev_info.ghost_distance > info.ghost_distance:
        #         reward -= 3
        #     elif self.prev_info.ghost_distance < info.ghost_distance:
        #         reward += 3
        #     if invalid_in_maze:
        #         reward -= 3
        #     return reward            
        if self.prev_info.food_distance >= info.food_distance and info.food_distance != -1:
            reward += 4
        elif self.prev_info.food_distance < info.food_distance and info.food_distance != -1:
            reward -= 3
        # if info.scared_ghost_distance <= 10 and self.prev_info.scared_ghost_distance >= info.scared_ghost_distance and info.scared_ghost_distance != -1:
        #     reward += 4
        # if not (info.ghost_distance >=1 and info.ghost_distance < 5):
        #     if action == REVERSED[self.last_action] and not info.invalid_move:
        #         reward -= 2
        if invalid_in_maze:
            reward -= 8
        else:
            if action == REVERSED[self.last_action] and hit_ghost or info.food_distance == -1:
                reward -= 4
        if not info.in_portal and info.food_distance == -1 and not hit_ghost:
            reward -= 20
        reward -= 1
        #assert(reward >=-30 and reward <= 30)
        self.writer.add_scalar('rewards', reward, global_step=self.steps)
        return reward
    def get_neighbors(self,info,action):
        row_indices, col_indices = np.where(info.frame == 5)
        invalid_in_maze = False
        if row_indices.size > 0:
            x = row_indices[0]
            y = col_indices[0]
            try:
                upper_cell = info.frame[x - 1][y]
                lower_cell = info.frame[x + 1][y]
                right_cell = info.frame[x][y + 1]
                left_cell = info.frame[x][y - 1]
            except IndexError:
                upper_cell = 0
                lower_cell = 0
                right_cell = 0
                left_cell = 0
            if info.invalid_move:
                if action == 0:
                    if upper_cell == 1:
                        invalid_in_maze=True
                elif action == 1:
                    if lower_cell == 1:
                        invalid_in_maze=True
                elif action == 2:
                    if left_cell == 1:
                        invalid_in_maze=True
                elif action == 3:
                    if right_cell == 1:
                        invalid_in_maze=True
        return invalid_in_maze
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
            while action == REVERSED[self.last_action] or self.game.get_invalid_action(action):
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
            torch.save(self.optimizer.state_dict(), os.path.join(
                os.getcwd() + "\\results", f"optimizer-{self.episode}-{self.steps}.pt"))

    def load_model(self, name, eval=False):
        path = os.path.join(os.getcwd() + "\\results", f"target-model-{name}.pt")
        self.target.load_state_dict(torch.load(path))
        path = os.path.join(os.getcwd() + "\\results", f"policy-model-{name}.pt")
        self.policy.load_state_dict(torch.load(path))
        path = os.path.join(
            os.getcwd() + "\\results", f"optimizer-{name}.pt")
        self.optimizer.load_state_dict(torch.load(path))
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
    def pacman_pos(self,state):
        index = np.where(state != 0)
        if len(index[0]) != 0:
            x = index[0][0]
            y = index[1][0]
            return (x,y)
        return None
    def train(self):
        if self.steps >= MAX_STEPS:
            self.save_model(force=True)
            exit()
        self.save_model()
        obs = self.game.start()
        self.episode += 1
        random_action = random.choice([0, 1, 2, 3])
        # obs, self.score, done, info = self.game.step(random_action)
        #state = self.process_state(obs)
        #state = torch.tensor(obs).float().to(device)
        for i in range(4):
            obs, self.score, done, info = self.game.step(random_action)
            self.buffer.append(info.frame)
        state = self.process_state(self.buffer)
        last_score = 0
        lives = 3
        while True:
            action = self.act(state)
            action_t = action.item()
            pacman_pos = self.pacman_pos(obs[2])
            counter = 0
            while True:
                    if not done:
                        obs, self.score, done, info = self.game.step(
                            action_t)
                        counter += 1
                        pacman_pos_new = self.pacman_pos(obs[2])
                        if counter > 40:
                            print("something went wrong")
                            break
                        if pacman_pos_new != pacman_pos or lives != info.lives or self.get_neighbors(info,action_t):
                            pacman_pos = pacman_pos_new
                            break
                    else:
                        break
            self.buffer.append(info.frame)
            hit_ghost = False
            if lives != info.lives:
                # self.write_matrix(self.buffer)
                hit_ghost = True
                lives -= 1
                if not done:
                    for i in range(3):
                        _, _, _, _ = self.game.step(action_t)
            #next_state = torch.tensor(obs).float().to(device)
            next_state = self.process_state(self.buffer)
            #next_state = self.process_state(obs)
            self.prev_info = info
            reward_ = self.get_reward(done, lives, hit_ghost, action_t, last_score, info)
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
            # if self.steps % 100000 == 0:
            #     self.scheduler.step()
            self.last_action = action_t
            if done:
                self.log()
                self.rewards.append(self.score)
                self.plot_rewards(items= self.rewards, avg=50)
                #self.plot_rewards(items = self.losses,label="losses",name="losses.png", avg=50)
                time.sleep(1)
                self.game.restart()
                torch.cuda.empty_cache()
                self.writer.add_scalar('episode reward', self.score, global_step=self.episode)
                break

    def log(self):
        # current_lr = self.optimizer.param_groups[0]["lr"]
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
    #agent.load_model(name="400-195112", eval=False)
    agent.rewards = []
    while True:
        agent.train()
        #agent.test()
