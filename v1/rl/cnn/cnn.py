import argparse
from json import load
from v1.game.achtung_process import AchtungProcess
import numpy as np
from itertools import count
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

from v1.game.config import WINDOW_HEIGHT, WINDOW_WIDTH

# parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor (default: 0.99)')
# parser.add_argument('--seed', type=int, default=543, metavar='N',
#                     help='random seed (default: 543)')
# parser.add_argument('--render', action='store_true',
#                     help='render the environment')
# parser.add_argument('--log-interval', type=int, default=1, metavar='N',
#                     help='interval between training status logs (default: 10)')
# parser.add_argument('--save-interval', type=int, default=1, metavar='N',
#                     help='interval between training status logs (default: 1000)')
# args = parser.parse_args()

class CNN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.softmax = nn.Softmax(dim=-1)

     # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        x = self.softmax(x)
        return x
  
class Policy():
    def __init__(self, height = 400, width = 400, c = 3, na = 3, gamma = 0.99, policy_net_name = None):
        super(Policy, self).__init__()
        
        self.gamma = gamma
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        if policy_net_name is None:
            self.net = CNN(height, width, c, na)
        else:
            self.net = self.load(policy_net_name)
        sp = self.net.parameters()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1.0e-5)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9)
        self.eps = np.finfo(np.float32).eps.item()

    def dump(self, model_file):
        torch.save(self.net, model_file)

    def load(self, policy_net_name):
        print("loading saved network")
        return torch.load(str(policy_net_name) + "0.ptmodel")


    def update_policy(self):
        R = 0
        rewards = []
        
        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0,R)
            
        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        # Calculate loss
        loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))
        
        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #Save and intialize episode history counters
        self.loss_history.append(loss.item())
        self.reward_history.append(np.sum(self.reward_episode))
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode= []

    def predict(self, state, saving:bool = True):
        state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0) 
        probs = self.net(state)
        m = Categorical(probs)
        action = m.sample()
        if saving:
            self.policy_history = torch.cat([self.policy_history, m.log_prob(action)])
        return action.item()



class CNN_Model():
    def __init__(self, batch_size = 100, min_reward = 250, policy_name=None):
        
        self.policy = Policy(height=WINDOW_HEIGHT,width=WINDOW_WIDTH,c=3,na=3,policy_net_name=policy_name)
        self.env = AchtungProcess(n=1, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
        self.batch_size = batch_size
        self.min_reward = min_reward

        self.env.env.speed = 0 # set to zero for training (i.e., no frame delay)
        self.env.env.render_game = False
        obs = self.env.reset()


    def learn(self, total_timesteps:int, batch_size = 1, render = False):
        running_reward = 10.0
        episode_length = []
        i_episode = 0

        for i in range(total_timesteps):
            for k in range(batch_size):
                print("episode:", len(episode_length))
                state = self.env.reset()
                ep_reward = 0
                for t in range(1, 1000):  # Don't infinite loop while learning
                    action = self.policy.predict(state)
                    state, reward, done, _ = self.env.step(action)
                    if render:
                        self.env.render()
                    self.policy.reward_episode.append(reward)
                    ep_reward += reward
                    if done:
                        break
                # Used to determine when the environment is solved.
                episode_length.append(t)
                running_reward = (running_reward * 0.95) + (ep_reward * 0.05)
                print('   Episode reward:', ep_reward, " steps:", t)
                print('   Running reward:', running_reward)
                i_episode += 1
            print("update policy")
            self.policy.update_policy()

            # if running_reward > self.min_reward:
            #     print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
            #     break

            # if (i_episode % 1000 == 0 or i_episode % 1000-1 == 0 or i_episode % 1000-2 == 0):
            #     np.save("images/ep" + str(i_episode) + "_" + str(t), state)

            # if (i_episode % 10 == 0):
            #     print('\n Saving checkpoint ' + "net_" + str(i_episode) + ".ptmodel\n")
            #     self.policy.dump("models/cnn_achtung0.ptmodel")
            #     with open("models/loss.txt", "wb") as f:   
            #         pickle.dump(self.policy.loss_history, f)
            #     with open("models/reward.txt", "wb") as f:   
            #         pickle.dump(self.policy.reward_history, f)
            #     with open("models/length.txt", "wb") as f:   
            #         pickle.dump(episode_length, f)     
    def predict(self, observations=None, state=None, deterministic=None):
        if observations.shape[0] == 1:
            observations = observations[0]
        
        action = self.policy.predict(observations, saving=False)
        # state, reward, done, _ = self.env.step(action)
        return [action], state

    def save(self, path):
        self.policy.dump(path)

# y = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
# print(y.shape)
# model = CNN(80,80,4,2)
# model.forward(y)


def get_cnn_model():
    # model = CNN_Model(batch_size=100, min_reward=250, policy_name="models/cnn_achtung")
    model = CNN_Model(batch_size=100, min_reward=250)
    return model

            
if __name__ == '__main__':
    # main()
    model = get_cnn_model()
    model.train()