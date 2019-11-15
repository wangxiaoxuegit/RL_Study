
import numpy as np
import gym
from rl_agent import DQNAgent

EPISODES = 50
batch_size = 32

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(action_size, state_size)
agent.load("./save/cartpole-ddqn.h5")
done = False
learncnt = 0

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    # while True:
    for time in range(500):
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.learn()
            learncnt += 1
        if learncnt % 200 == 0:
            agent.update_target_model()
        if done:
            agent.update_target_model()
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e+1, EPISODES, time, agent.epsilon))
            break
    if e % 10 == 0:
        agent.save("./save/cartpole-ddqn.h5")
