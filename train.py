import torch
import numpy as numpy
import gfootball.env as football_env
from network import Agent
import matplotlib.pyplot as plt
#from utils import plotLearning


if __name__ == "__main__":
    # hyperparameters
    state_dim = 115
    # no. of possible actions
    action_dim = 19
    alpha= 0.0001
    beta = 0.0005
    gamma = .99
    # number of episodes
    n_episodes = 7000
    # instantiate the actor 
    agent = Agent(state_dim,action_dim,alpha,beta,gamma)
    display_train = False
    # render
    if display_train:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
    else:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115',render=True)

    score_history = []
    # start the episodes
    for i in range(n_episodes):
        done = False
        score = 0
        # reset
        observation = env.reset()
        while not done:
            # actor predicting the actions
            action = agent.action(observation)
            observation_ , reward, done, info = env.step(action)
            score +=reward
            # upadte the weights using critic
            agent.update(observation,reward,observation_,done)
            observation = observation_
        print ('episode',i,'score %.3f' %score)
        score_history.append(score)
        if (i%1000) == 0 : 
            torch.save(agent.actor, './PPO_act{}.pth'.format(i))
            #torch.save(agent.critic, './PPO_cri{}.pth'.format(i))
