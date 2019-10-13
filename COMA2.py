from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from env_FindGoals import EnvFindGoals
import matplotlib.pyplot as plt

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Actor(nn.Module):
    def __init__(self, N_action):
        super(Actor, self).__init__()
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1 = Flatten()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, self.N_action)

    def get_action(self, h):
        h1 = F.relu(self.conv1(h))
        h1 = self.flat1(h1)
        h1 = F.relu(self.fc1(h1))
        h = F.softmax(self.fc2(h1), dim=1)
        m = Categorical(h.squeeze(0))
        return m.sample().item(), h

class Critic(nn.Module):
    def __init__(self, N_action):
        super(Critic, self).__init__()
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1 = Flatten()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat2 = Flatten()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, N_action*N_action)

    def get_value(self, s1, s2):
        h1 = F.relu(self.conv1(s1))
        h1 = self.flat1(h1)
        h2 = F.relu(self.conv2(s2))
        h2 = self.flat2(h2)
        h = torch.cat([h1, h2], 1)
        x = F.relu(self.fc1(h))
        x = self.fc2(x)
        return x

class COMA(object):
    def __init__(self, N_action):
        self.N_action = N_action
        self.actor1 = Actor(self.N_action)
        self.actor2 = Actor(self.N_action)
        self.critic = Critic(self.N_action)
        self.gamma = 0.95
        self.c_loss_fn = torch.nn.MSELoss()

    def get_action(self, obs1, obs2):
        action1, pi_a1 = self.actor1.get_action(self.img_to_tensor(obs1).unsqueeze(0))
        action2, pi_a2 = self.actor2.get_action(self.img_to_tensor(obs2).unsqueeze(0))
        return action1, pi_a1, action2, pi_a2

    def img_to_tensor(self, img):
        img_tensor = torch.FloatTensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def cross_prod(self, pi_a1, pi_a2):
        new_pi = torch.zeros(1, self.N_action*self.N_action)
        for i in range(self.N_action):
            for j in range(self.N_action):
                new_pi[0, i*self.N_action+j] = pi_a1[0, i]*pi_a2[0, j]
        return new_pi

    def train(self, o1_list, a1_list, pi_a1_list, o2_list, a2_list, pi_a2_list, r_list):
        a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=3e-4)
        a2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=3e-4)
        c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        T = len(r_list)
        obs1 = self.img_to_tensor(o1_list[0]).unsqueeze(0)
        obs2 = self.img_to_tensor(o2_list[0]).unsqueeze(0)
        for t in range(1, T):
            temp_obs1 = self.img_to_tensor(o1_list[t]).unsqueeze(0)
            obs1 = torch.cat([obs1, temp_obs1], dim=0)
            temp_obs2 = self.img_to_tensor(o2_list[t]).unsqueeze(0)
            obs2 = torch.cat([obs2, temp_obs2], dim=0)

        Q = self.critic.get_value(obs1, obs2)
        Q_est = Q.clone()
        for t in range(T - 1):
            a_index = a1_list[t]*self.N_action + a2_list[t]
            Q_est[t][a_index] = r_list[t] + self.gamma * torch.sum(self.cross_prod(pi_a1_list[t+1], pi_a2_list[t+1])*Q_est[t+1, :])
        a_index = a1_list[T - 1] * self.N_action + a2_list[T - 1]
        Q_est[T - 1][a_index] = r_list[T - 1]
        c_loss = self.c_loss_fn(Q, Q_est.detach())
        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step()

        A1_list = []
        for t in range(T):
            temp_Q1 = torch.zeros(1, self.N_action)
            for a1 in range(self.N_action):
                temp_Q1[0, a1] = Q[t][a1*self.N_action + a2_list[t]]
            a_index = a1_list[t] * self.N_action + a2_list[t]
            temp_A1 = Q[t, a_index] - torch.sum(pi_a1_list[t]*temp_Q1)
            A1_list.append(temp_A1)

        A2_list = []
        for t in range(T):
            temp_Q2 = torch.zeros(1, self.N_action)
            for a2 in range(self.N_action):
                temp_Q2[0, a2] = Q[t][a1_list[t] * self.N_action + a2]
            a_index = a1_list[t] * self.N_action + a2_list[t]
            temp_A2 = Q[t, a_index] - torch.sum(pi_a2_list[t] * temp_Q2)
            A2_list.append(temp_A2)

        a1_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a1_loss = a1_loss + A1_list[t].item() * torch.log(pi_a1_list[t][0, a1_list[t]])
        a1_loss = -a1_loss / T
        a1_optimizer.zero_grad()
        a1_loss.backward()
        a1_optimizer.step()

        a2_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a2_loss = a2_loss + A2_list[t].item() * torch.log(pi_a2_list[t][0, a2_list[t]])
        a2_loss = -a2_loss / T
        a2_optimizer.zero_grad()
        a2_loss.backward()
        a2_optimizer.step()

if __name__ == '__main__':
    torch.set_num_threads(1)
    env = EnvFindGoals()
    max_epi_iter = 1000
    max_MC_iter = 200
    agent = COMA(N_action=5)
    train_curve = []
    for epi_iter in range(max_epi_iter):
        env.reset()
        o1_list = []
        a1_list = []
        pi_a1_list = []
        o2_list = []
        a2_list = []
        pi_a2_list = []
        r_list = []
        acc_r = 0
        for MC_iter in range(max_MC_iter):
            # env.render()
            obs1 = env.get_agt1_obs()
            obs2 = env.get_agt2_obs()
            o1_list.append(obs1)
            o2_list.append(obs2)
            action1, pi_a1, action2, pi_a2 = agent.get_action(obs1, obs2)
            a1_list.append(action1)
            pi_a1_list.append(pi_a1)
            a2_list.append(action2)
            pi_a2_list.append(pi_a2)
            [reward_1, reward_2], done = env.step([action1, action2])
            acc_r = acc_r + reward_1
            r_list.append(reward_1)
            if done:
                break
        if epi_iter % 10 == 0:
            train_curve.append(acc_r/MC_iter)
        print('Episode', epi_iter, 'reward', acc_r/MC_iter)
        agent.train(o1_list, a1_list, pi_a1_list, o2_list, a2_list, pi_a2_list, r_list)
    plt.plot(train_curve, linewidth=1, label='COMA')
    plt.show()