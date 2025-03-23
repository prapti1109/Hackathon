import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf

########################################################################
# 1. Data Download & Preprocessing
########################################################################

ticker = "TSLA"
start_date = "2018-01-01"
end_date = "2022-01-01"
df = yf.download(ticker, start=start_date, end=end_date)

df["Return"] = df["Close"].pct_change().fillna(0.0)
df["Target"] = df["Close"].shift(-1)
df = df.dropna()

# Dummy sentiment scores (replace with real sentiment extraction)
np.random.seed(42)
sentiment_array = np.random.normal(0, 1, len(df))
df["Sentiment"] = sentiment_array

# Feature columns
features = df[["Close", "Volume", "Return", "Sentiment"]].values
targets = df["Target"].values

window_size = 5
feature_dim = features.shape[1]

train_ratio = 0.8
train_size = int(len(df) * train_ratio)

train_features = features[:train_size]
train_targets  = targets[:train_size]
test_features  = features[train_size:]
test_targets   = targets[train_size:]

########################################################################
# 2. Custom Environment (Gym-Like)
########################################################################

class PricePredictionEnv:
    def __init__(self, features, targets, window_size=5):
        self.features = features
        self.targets = targets
        self.window_size = window_size
        self.current_step = 0
        self.max_step = len(self.features) - window_size - 1

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        start = self.current_step
        end = self.current_step + self.window_size
        return self.features[start:end]

    def step(self, action):
        real_price = self.targets[self.current_step + self.window_size]
        error = abs(action - real_price)
        reward = -error

        self.current_step += 1
        done = (self.current_step + self.window_size >= len(self.features))
        next_state = None if done else self._get_state()
        return next_state, reward, done

########################################################################
# 3. Actor-Critic (PPO) Network
########################################################################

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.actor_mean(x)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value

########################################################################
# 4. PPO Agent
########################################################################

class PPOAgent:
    def __init__(self, state_shape, lr=1e-3, gamma=0.99, lam=0.95, clip_eps=0.2):
        flat_input_dim = state_shape[0] * state_shape[1]
        self.net = ActorCritic(flat_input_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state):
        s = torch.FloatTensor(state.flatten()).unsqueeze(0)
        mean, std, val = self.net(s)
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.sample()
        log_prob = dist.log_prob(action_sample)
        return action_sample.item(), log_prob.item(), val.item()

    def store_transition(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def finish_trajectory(self, last_val=0):
        advantages = []
        rewards_to_go = []
        gae = 0
        running_return = 0

        for t in reversed(range(len(self.rewards))):
            next_value = 0 if self.dones[t] else (self.values[t+1] if t < len(self.values)-1 else last_val)
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta + self.gamma * self.lam * (0 if self.dones[t] else gae)
            advantages.append(gae)
            running_return = self.rewards[t] + self.gamma * (0 if self.dones[t] else running_return)
            rewards_to_go.append(running_return)

        advantages.reverse()
        rewards_to_go.reverse()
        return torch.FloatTensor(advantages), torch.FloatTensor(rewards_to_go)

    def update(self):
        advantages, returns = self.finish_trajectory()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        s = torch.FloatTensor([st.flatten() for st in self.states])
        a = torch.FloatTensor(self.actions).unsqueeze(-1)
        lp_old = torch.FloatTensor(self.log_probs).unsqueeze(-1)
        v_old = torch.FloatTensor(self.values).unsqueeze(-1)

        mean, std, vals = self.net(s)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(a)

        ratio = (new_log_probs - lp_old).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(vals, returns.unsqueeze(-1))
        total_loss = actor_loss + 0.5 * critic_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

########################################################################
# 5. Training
########################################################################

env = PricePredictionEnv(train_features, train_targets, window_size)
agent = PPOAgent(state_shape=(window_size, feature_dim), lr=1e-3)

episodes = 10
for ep in range(episodes):
    state = env.reset()
    ep_reward = 0
    while True:
        act, logp, val = agent.select_action(state)
        nxt, rew, done = env.step(act)
        agent.store_transition(state, act, rew, val, logp, done)
        ep_reward += rew
        if done:
            agent.update()
            break
        state = nxt
    print(f"Episode {ep+1}/{episodes}, Reward: {ep_reward:.2f}")

########################################################################
# 6. Testing / Evaluation
########################################################################

test_env = PricePredictionEnv(test_features, test_targets, window_size)
test_state = test_env.reset()
test_reward = 0
step_count = 0
while True:
    act, _, _ = agent.select_action(test_state)
    nxt, rew, done = test_env.step(act)
    test_reward += rew
    step_count += 1
    if done:
        break
    test_state = nxt

print(f"Test total reward: {test_reward:.2f} over {step_count} steps.")
