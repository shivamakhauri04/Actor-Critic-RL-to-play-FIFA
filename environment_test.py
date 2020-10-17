import gfootball.env as football_env


def random():
    # render the environment
    display_train = False
    if display_train:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
    else:
        env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115',render=True)

    total_rewards = 0.0
    total_steps = 0
    obs = env.reset()
    # start an episode. 
    while True:
        action = env.action_space.sample()
        # choose a random action
        obs, reward, done, _ = env.step(action)
        # obtain rewards
        total_rewards += reward
        total_steps += 1
        if done:
            break
    print ("episode done %d steps. total rewards %.2f" %(total_steps,total_rewards))



if __name__ == "__main__":
    random()



