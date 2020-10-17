import torch
import torch.nn.functional as F
import numpy as numpy
import gfootball.env as football_env
from network import Agent
import matplotlib.pyplot as plt
#from utils import plotLearning


if __name__ == "__main__":
    # hyperparameters
    state_dim = 115
    # number of possible actions
    action_dim = 19
    # retention rate for actor and critic
    alpha= 0.0001
    beta = 0.0005
    gamma = .95
    # number of episodes to test
    n_episodes = 20
    

    # instantiate the actor
    agent = Agent(state_dim,action_dim,alpha,beta,gamma)
    # load the trained model
    agent.actor = torch.load('PPO_act6000.pth')
    display_train = False
    # render environment
    if display_train:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
    else:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115',render=True)
    # append the scores
    score_history = []
    # testing pipeline
    with torch.no_grad():
        for i in range(n_episodes):
            done = False
            score = 0
            observation = env.reset()
            while not done:
                # obtain action from actor
                probablities = F.softmax(agent.actor.forward(observation))
                action_probs = torch.distributions.Categorical(probablities)
                action = action_probs.sample()
                agent.log_probs = action_probs.log_prob(action)
                action = action.item()
                print (action)
                # Obtain new state and rewards
                observation_ , reward, done, info = env.step(action)
                score +=reward
                observation = observation_
            print ('episode',i,'score %.3f' %score)
            score_history.append(score)


