import gfootball.env as football_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal


# Common network for the actor and critic
class Network(nn.Module):
    def __init__(self,lr,state_dim,num_actions):
        super(Network,self).__init__()
        # learning rate
        self.lr = lr
        # input size
        self.input_dim = state_dim
        # flly connected layers
        self.fc1_dims = 64
        self.fc2_dims = 32
        # number of possible actions
        self.num_actions = num_actions
        # fully connected layers
        self.fc1 = nn.Linear(self.input_dim,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.num_actions)
        # optimizer for backpropagation
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)
    # forward pass for the neural network
    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = nn.functional.tanh(self.fc1(state))
        x = nn.functional.tanh(self.fc2(x))
        x = nn.functional.tanh(self.fc3(x))
        return x


class Agent():
    def __init__(self,state_dim,action_dim,alpha,beta,gamma):
        self.log_probs =  None
        # retention rate
        self.gamma = gamma
        # Actor network for action prediction
        self.actor = Network(alpha,state_dim,action_dim)
        # critic network for value prediction
        self.critic = Network(beta,state_dim,1)

    def action(self,observation):
        # actor being used
        probablities = F.softmax(self.actor.forward(observation))
        action_probs = torch.distributions.Categorical(probablities)
        action = action_probs.sample()
        # find the probablities for actions
        self.log_probs = action_probs.log_prob(action)
        return action.item()
        

    def update(self, state, reward, new_state, done):
        # backpropagation
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        # critic being used
        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)
        # loss function for actor critic
        delta = (reward + self.gamma*critic_value_ - critic_value)
        actor_loss = -self.log_probs *delta
        critic_loss = delta**2
        # backpropagate on combined loss
        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        




        


def main():
    display_train = False
    # render environment
    if display_train:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
    else:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115',render=True)
    
    print ("action space=",env.action_space)
    print ("observation space",env.observation_space)



