import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# 定义DQNAgent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(10000)
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values, dim=1).item()
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# 训练函数
def train(agent, env, test_interval=100):
    episode = 0
    while True:
        episode += 1
        state, _ = env.reset()
        total_reward = 0
        done = False
        max_steps = 500
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break
        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {step+1}")
        if episode % test_interval == 0:
            # 保存当前epsilon值
            original_epsilon = agent.epsilon
            # 设置epsilon为0以禁用探索
            agent.epsilon = 0.0
            # 进行测试并计算平均奖励
            test_reward = test(agent, env, episodes=10, render=False)
            # 创建一个新的环境实例来渲染展示
            test_env = gym.make('CartPole-v1', render_mode="human")
            test(agent, test_env, episodes=1, render=True)
            # 关闭渲染环境
            test_env.close()
            # 恢复原始epsilon值
            agent.epsilon = original_epsilon
            print(f"Test after {episode} episodes, Average Reward: {test_reward}")

# 测试函数
def test(agent, env, episodes=10, render=False):
    total_rewards = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_rewards += reward
    average_reward = total_rewards / episodes
    return average_reward

# 主函数
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    train(agent, env, test_interval=100)