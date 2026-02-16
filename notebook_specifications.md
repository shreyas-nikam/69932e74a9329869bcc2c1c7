
# Dynamic Asset Allocation via Reinforcement Learning: An Adaptive Strategy for Non-Stationary Markets

## Case Study Introduction: Alpha Capital's Adaptive Edge

**Persona:** Alexandra Chen, CFA Charterholder and Senior Portfolio Manager at Alpha Capital Management.  
**Organization:** Alpha Capital Management, an investment firm specializing in quantitative strategies for high-net-worth individuals and institutional clients.

Alexandra faces a persistent challenge: traditional static asset allocation models (like a fixed 60/40 stock-bond split or periodically rebalanced Markowitz portfolios) often underperform during rapidly changing market regimes. The financial landscape is increasingly non-stationary, with sudden shifts in volatility, correlations, and return profiles (e.g., the COVID-19 crash, 2022 rate hikes). Her goal is to explore how an adaptive, AI-driven policy can dynamically rebalance a multi-asset portfolio to achieve superior risk-adjusted returns while realistically accounting for transaction costs. She needs a robust framework that can learn to adjust asset weights in real-time, anticipating market shifts rather than reacting slowly.

This notebook will guide Alexandra through building a Reinforcement Learning (RL) agent capable of dynamic asset allocation. She will define a multi-asset investment environment, train a cutting-edge RL algorithm (PPO) to learn an optimal allocation policy, and rigorously compare its performance against several traditional benchmarks. The ultimate aim is to equip Alpha Capital with a more resilient and adaptive portfolio management tool.

---

## 1. Setting Up the Dynamic Portfolio Lab

Alexandra begins by setting up her Python environment, installing the necessary libraries, and importing them. This ensures her workspace is ready for data acquisition, RL environment creation, model training, and performance analysis.

### Install Required Libraries

Before diving into the analysis, Alexandra installs all the necessary Python packages. This ensures she has access to tools for financial data, numerical operations, reinforcement learning, and visualization.

```python
!pip install yfinance pandas numpy gymnasium stable-baselines3 matplotlib seaborn
```

### Import Required Dependencies

Alexandra now imports the libraries she will use throughout her workflow. This includes `yfinance` for market data, `pandas` for data manipulation, `numpy` for numerical operations, `gymnasium` for creating the RL environment, `stable_baselines3` for the PPO agent, and `matplotlib` and `seaborn` for visualizations.

```python
import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("All required libraries imported successfully.")
```

---

## 2. Acquiring Market Data and Engineering Features

To build an intelligent allocation policy, Alexandra needs rich market data that captures the dynamics of her chosen asset classes. She will download historical prices for a diversified portfolio and then engineer relevant features that the RL agent can use to understand market conditions.

### Story + Context + Real-World Relevance

Alexandra's multi-asset portfolio includes:
*   **SPY:** S&P 500 ETF (Equities)
*   **TLT:** iShares 20+ Year Treasury Bond ETF (Long Bonds)
*   **GLD:** SPDR Gold Shares (Gold)
*   **SHY:** iShares 1-3 Year Treasury Bond ETF (Cash proxy / Short-Term Bonds)

These assets represent common building blocks in diversified portfolios. For her RL agent to make informed decisions, it needs to observe not just raw returns but also indicators of market state, such as short-term momentum, volatility, and cross-asset correlations. These features help the agent understand changing market regimes (e.g., risk-on vs. risk-off).

The dataset is split into a training period (2010-01-01 to 2019-12-31) for the RL agent to learn from, and a separate out-of-sample test period (2020-01-01 to 2024-12-31) for unbiased performance evaluation, which includes major market events like the COVID crash and the 2022 rate shock.

```python
def load_and_prepare_data(tickers_dict, start_date, end_date):
    """
    Downloads historical 'Adj Close' prices, calculates daily returns,
    and engineers market features for the RL environment.
    
    Args:
        tickers_dict (dict): A dictionary mapping ticker symbols to asset names.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).
        
    Returns:
        tuple: (pd.DataFrame, pd.DataFrame) of returns and engineered features.
    """
    print(f"Downloading data for {list(tickers_dict.keys())} from {start_date} to {end_date}...")
    data = yf.download(list(tickers_dict.keys()), start=start_date, end=end_date)['Adj Close']
    data.columns = [tickers_dict[c] for c in data.columns]
    returns = data.pct_change().dropna()

    features = pd.DataFrame(index=returns.index)
    for asset in returns.columns:
        # 5-day rolling mean returns (momentum)
        features[f'{asset}_ret_5d'] = returns[asset].rolling(5).mean()
        # 21-day rolling standard deviation (volatility), annualized
        features[f'{asset}_vol_21d'] = returns[asset].rolling(21).std() * np.sqrt(252)

    # 63-day rolling cross-correlation mean across all assets
    # Measures overall market interconnectedness or risk appetite
    features['cross_corr_mean_63d'] = returns.rolling(63).corr().mean(axis=1)

    # Drop any rows with NaN values resulting from rolling calculations
    features = features.dropna()
    returns = returns.loc[features.index] # Align returns with features index

    print(f"\nData loaded and features engineered. Shape: {features.shape}")
    print(f"Training period: {features.index[0]} to 2019-12-31")
    print(f"Test period: 2020-01-01 to {features.index[-1]}")
    print(f"Assets: {list(returns.columns)}")
    print(f"Features per step: {features.shape[1]}")
    
    return returns, features

# Define asset universe and date range
ASSET_TICKERS = {'SPY': 'Equities', 'TLT': 'Long Bonds', 'GLD': 'Gold', 'SHY': 'Cash'}
START_DATE = '2010-01-01'
END_DATE = '2024-12-31' # Inclusive of 2024 for the test set

# Execute data loading and feature engineering
all_returns, all_features = load_and_prepare_data(ASSET_TICKERS, START_DATE, END_DATE)

# Define train and test masks based on features index for consistency
TRAIN_END_DATE = '2019-12-31'
train_mask = all_features.index <= pd.to_datetime(TRAIN_END_DATE)
test_mask = all_features.index > pd.to_datetime(TRAIN_END_DATE)

train_returns = all_returns[train_mask]
train_features = all_features[train_mask]

test_returns = all_returns[test_mask]
test_features = all_features[test_mask]

print(f"\nTraining data shape (returns): {train_returns.shape}, (features): {train_features.shape}")
print(f"Testing data shape (returns): {test_returns.shape}, (features): {test_features.shape}")
```

### Explanation of Execution

Alexandra reviews the output, ensuring the data spans the correct periods and that the engineered features are present. The distinct training and testing periods are crucial for an honest out-of-sample evaluation, preventing look-ahead bias and assessing the true adaptiveness of the RL agent. The market features she created, such as rolling mean returns, volatility, and cross-correlations, will serve as the agent's "eyes" into the market state.

---

## 3. Designing the Multi-Asset Portfolio Reinforcement Learning Environment

For the RL agent to learn, Alexandra must define the rules of its world: the portfolio environment. This involves specifying the state observations, the available actions (asset weights), the dynamics of portfolio value, and critically, how rewards are calculated based on risk-adjusted performance and transaction costs.

### Story + Context + Real-World Relevance

Alexandra needs to translate the complex world of portfolio management into a `gymnasium` environment. This is a standard practice in RL, allowing her to simulate market interactions. The agent will learn by interacting with this environment, receiving observations and rewards after each allocation decision.

Key components of her environment design include:
*   **State:** The agent needs to know its current portfolio allocation and the latest market features.
*   **Action Space:** The agent decides how to allocate capital across the assets. This must be a continuous action space (asset weights sum to 1), requiring a policy gradient method like PPO. The softmax function ensures valid portfolio weights from raw agent outputs:
    $$ w_i = \frac{e^{a_i}}{\sum_{j=1}^{K} e^{a_j}} $$
    where $w_i$ is the weight of asset $i$, $a_i$ is the raw action output for asset $i$, and $K$ is the number of assets. This ensures $w_i > 0$ and $\sum w_i = 1$.
*   **Portfolio Value Dynamics:** The portfolio value ($V_t$) evolves based on asset returns and is penalized by transaction costs ($TC_t$):
    $$ V_{t+1} = V_t \cdot (1 + \mathbf{w}_t^{\top} \mathbf{r}_{t+1}) - TC_t $$
    where $\mathbf{w}_t$ is the weight vector at time $t$, $\mathbf{r}_{t+1}$ is the vector of asset returns at $t+1$, and the transaction cost $TC_t$ is calculated based on turnover and a per-unit transaction rate:
    $$ TC_t = c \cdot ||\mathbf{w}_{t+1} - \mathbf{w}_t||_1 \cdot V_t $$
    Here, $c$ is the per-unit transaction cost rate (e.g., 0.001 for 10 bps), and $||\cdot||_1$ is the L1 norm, measuring total turnover.
*   **Reward Function (Utility-Based):** The reward function is crucial as it encodes Alexandra's investment policy statement (IPS) and risk preferences. It guides the agent to maximize risk-adjusted returns. She opts for the **Mean-Variance Utility** as the default, a direct connection to classical CFA utility theory ($U = E[r] - \frac{1}{2}A \sigma^2$):
    $$ R_t = r_{p,t} - \lambda \sigma^2_{p,t} $$
    where $r_{p,t}$ is the portfolio return at time $t$, $\sigma^2_{p,t}$ is the rolling 21-day annualized portfolio variance, and $\lambda$ is the risk aversion coefficient. A higher $\lambda$ indicates greater risk aversion, leading the agent to adopt more conservative strategies. This parameter is directly analogous to the risk aversion in a traditional IPS.

Alexandra also considers two other common utility functions for different investor profiles, though Mean-Variance will be the default for this lab:
    *   **Sortino Reward (downside-only):** Penalizes drawdowns but not upside volatility.
        $$ R_t = r_{p,t} - \lambda \cdot DD^+_t $$
        where $DD^+_t = \max(0, V_{\text{peak}} - V_t) / V_{\text{peak}}$ is the current drawdown relative to the peak.
    *   **Log-Wealth (Kelly-like):** Maximizes geometric growth rate, penalizing large bets.
        $$ R_t = \log(1 + r_{p,t}) $$

```python
class PortfolioAllocEnv(gym.Env):
    """
    Multi-asset portfolio allocation environment for Reinforcement Learning.
    State: current weights + market features (returns, vol, corr)
    Action: continuous weight vector (softmax-normalized to simplex)
    Reward: risk-adjusted portfolio return based on utility theory.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, returns_df, features_df, initial_value=100000,
                 tc_rate=0.001, risk_aversion=2.0, reward_type='mean_variance'):
        super().__init__()
        self.returns = returns_df.values
        self.features = features_df.values
        self.asset_names = returns_df.columns.tolist()
        self.n_assets = returns_df.shape[1]
        self.n_features_per_asset = (features_df.shape[1] - 1) // self.n_assets # -1 for cross_corr_mean
        self.initial_value = initial_value
        self.tc_rate = tc_rate
        self.risk_aversion = risk_aversion
        self.reward_type = reward_type
        self.n_steps = len(returns_df) - 1 # Number of trading days in environment

        # Continuous action space: raw weights (softmax-normalized)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation space: current weights + market features
        # Assuming features_df contains all features concatenated already
        obs_dim = self.n_assets + features_df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize environment state
        self.reset()

    def _softmax(self, x):
        """Converts raw action outputs to valid portfolio weights (sums to 1)."""
        e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return e_x / e_x.sum()

    def _get_obs(self):
        """Returns the current observation for the agent."""
        # Current portfolio weights + market features for the current step index
        return np.concatenate([
            self.weights,
            self.features[self.step_idx]
        ]).astype(np.float32)

    def _compute_reward(self, port_return):
        """
        Computes the utility-based reward for the current portfolio return.
        
        Args:
            port_return (float): The daily portfolio return.
            
        Returns:
            float: The calculated reward.
        """
        if self.reward_type == 'mean_variance':
            # Mean-Variance Utility: R_t = r_{p,t} - lambda * sigma^2_{p,t}
            # Need at least 21 days for rolling variance
            if len(self.returns_history) < 21:
                return port_return # No variance penalty initially
            
            # Use recent returns to calculate rolling variance
            recent_returns = np.array(self.returns_history[-21:])
            # Annualize variance: daily variance * 252 trading days
            portfolio_variance = np.var(recent_returns) * 252 
            reward = port_return - self.risk_aversion * portfolio_variance
            return reward

        elif self.reward_type == 'sortino':
            # Sortino Reward: R_t = r_{p,t} - lambda * DD^+_t
            current_value = self.portfolio_value
            peak_value = np.max(self.portfolio_value_history)
            drawdown = max(0, (peak_value - current_value) / peak_value)
            reward = port_return - self.risk_aversion * drawdown
            return reward

        elif self.reward_type == 'log_wealth':
            # Log-Wealth Reward: R_t = log(1 + r_{p,t})
            return np.log(1 + port_return)
            
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def step(self, action):
        """
        Executes one step in the environment given an action (new weights).
        
        Args:
            action (np.ndarray): Raw action output from the agent.
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # Normalize action to valid weights
        new_weights = self._softmax(action)

        # Calculate turnover and transaction costs
        turnover = np.sum(np.abs(new_weights - self.weights))
        tc = self.tc_rate * turnover * self.portfolio_value # TC based on current value before return

        # Get asset returns for the current step
        asset_returns = self.returns[self.step_idx]
        port_return = np.dot(new_weights, asset_returns)

        # Update portfolio value
        self.portfolio_value *= (1 + port_return)
        self.portfolio_value -= tc # Deduct transaction costs

        # Update weights and history
        self.weights = new_weights
        self.returns_history.append(port_return)
        self.portfolio_value_history.append(self.portfolio_value)

        # Compute reward
        reward = self._compute_reward(port_return)

        # Advance step
        self.step_idx += 1
        done = self.step_idx >= self.n_steps
        truncated = False # No truncation logic for this env, just done

        info = {
            'portfolio_value': self.portfolio_value,
            'weights': new_weights.copy(),
            'turnover': turnover,
            'port_return': port_return,
            'transaction_cost': tc
        }

        return self._get_obs(), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        
        Args:
            seed (int, optional): Seed for reproducibility.
            options (dict, optional): Additional options.
            
        Returns:
            tuple: (initial_observation, info)
        """
        super().reset(seed=seed)
        self.step_idx = 0
        self.portfolio_value = self.initial_value
        # Start with equal weights
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.returns_history = []
        self.portfolio_value_history = [self.initial_value] # Track portfolio value over time for drawdown/peak

        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy()
        }
        return self._get_obs(), info

# Define environment parameters
INITIAL_PORTFOLIO_VALUE = 100000
TRANSACTION_COST_RATE = 0.001 # 10 basis points per unit of turnover
RISK_AVERSION_COEFF = 2.0 # Lambda for Mean-Variance Utility
REWARD_FUNCTION_TYPE = 'mean_variance' # Options: 'mean_variance', 'sortino', 'log_wealth'

# Create the training environment instance
train_env_instance = PortfolioAllocEnv(
    returns_df=train_returns, 
    features_df=train_features, 
    initial_value=INITIAL_PORTFOLIO_VALUE, 
    tc_rate=TRANSACTION_COST_RATE, 
    risk_aversion=RISK_AVERSION_COEFF,
    reward_type=REWARD_FUNCTION_TYPE
)

# Wrap the environment in a DummyVecEnv for stable-baselines3 compatibility
vec_env_train = DummyVecEnv([lambda: train_env_instance])

print(f"Portfolio Allocation Environment created for training with {REWARD_FUNCTION_TYPE} reward and lambda={RISK_AVERSION_COEFF}.")
print(f"Observation Space Shape: {vec_env_train.observation_space.shape}")
print(f"Action Space Shape: {vec_env_train.action_space.shape}")
```

### Explanation of Execution

Alexandra has successfully defined her `PortfolioAllocEnv`. This environment now simulates the daily interactions of a portfolio manager: observing market conditions (state), making allocation decisions (action), seeing how the portfolio value changes, and receiving a reward based on her risk-adjusted return preferences. The explicit inclusion of transaction costs is critical; without it, the agent would likely rebalance excessively, leading to unrealistic performance. The `risk_aversion` parameter acts as a direct input to her Investment Policy Statement, allowing her to control the agent's risk appetite.

---

## 4. Training the Reinforcement Learning Agent with Proximal Policy Optimization (PPO)

With the environment defined, Alexandra can now train her RL agent. She will use Proximal Policy Optimization (PPO), a robust policy gradient algorithm well-suited for continuous action spaces like asset weight allocation.

### Story + Context + Real-World Relevance

Alexandra uses PPO because it's a state-of-the-art, stable, and widely used algorithm for continuous control problems, perfect for learning asset weights directly. Traditional Q-learning algorithms are less suitable as they require discretizing the action space (e.g., specific percentage steps), which can lead to a "curse of dimensionality" for multiple assets.

PPO works by optimizing a "clipped surrogate objective" function, which strikes a balance between making aggressive policy updates and ensuring stability. The objective function is defined as:
$$ L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right] $$
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio (how much the new policy $\pi_\theta$ is preferred over the old policy $\pi_{\theta_{\text{old}}}$ for action $a_t$ in state $s_t$), and $\hat{A}_t$ is the advantage estimate. The `clip` function limits the policy updates, preventing them from becoming too large and potentially destabilizing the learning process.

The advantage estimate $\hat{A}_t$ is typically calculated using Generalized Advantage Estimation (GAE):
$$ \hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} $$
where $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$ is the Temporal Difference (TD) error.
-   $\gamma$ (discount factor) determines how much future rewards are considered (e.g., 0.99 for long-term investors).
-   $\lambda$ (GAE parameter) balances bias and variance in the advantage estimate (e.g., 0.95 for moderate balance).

These parameters ($\gamma$ and $\lambda$) directly influence the agent's learning horizon and its aggressiveness in incorporating new information, which is relevant for a portfolio manager deciding how quickly to adapt to market shifts.

Alexandra will train the agent for a substantial number of timesteps (500,000) on her historical data.

```python
def train_ppo_agent(env, total_timesteps, model_save_path):
    """
    Trains a PPO agent on the given environment and saves the trained model.
    
    Args:
        env (DummyVecEnv): The wrapped RL environment.
        total_timesteps (int): Total timesteps for training.
        model_save_path (str): File path to save the trained model.
        
    Returns:
        PPO: The trained PPO model.
    """
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    model = PPO(
        "MlpPolicy",            # Multi-layer perceptron policy for continuous actions
        env,
        learning_rate=3e-4,     # Common learning rate for PPO
        n_steps=2048,           # Number of steps to run for each environment per update
        batch_size=64,          # Mini-batch size for policy and value updates
        n_epochs=10,            # Number of epochs to optimize the surrogate loss
        gamma=0.99,             # Discount factor (long-horizon investor)
        gae_lambda=0.95,        # GAE parameter (moderate bias-variance balance)
        clip_range=0.2,         # Clipping parameter for PPO's clipped surrogate objective
        ent_coef=0.01,          # Entropy coefficient for exploration-exploitation balance
        verbose=0,              # Set to 1 for training progress output, 0 for silent
        tensorboard_log="./ppo_portfolio_tensorboard/" # Path for TensorBoard logs
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_save_path)
    print(f"PPO training complete. Model saved to {model_save_path}")
    return model

# Define training parameters
TOTAL_TIMESTEPS = 500000
MODEL_SAVE_PATH = "ppo_portfolio_agent.zip"

# Execute PPO agent training
ppo_model = train_ppo_agent(vec_env_train, TOTAL_TIMESTEPS, MODEL_SAVE_PATH)
```

### Explanation of Execution

Alexandra observes that the PPO agent has completed its training over 500,000 timesteps. This process allows the agent to iteratively refine its allocation policy by observing market features, making decisions, and adjusting its strategy based on the risk-adjusted rewards it receives. The saved model (`ppo_portfolio_agent.zip`) now encapsulates the learned dynamic allocation policy, ready for evaluation against new, unseen market data.

---

## 5. Implementing Baseline Portfolio Strategies for Comparison

Before assessing the RL agent's performance, Alexandra needs robust benchmarks. She will implement four traditional portfolio strategies that a typical portfolio manager might use, allowing for an honest comparison of the RL agent's adaptive capabilities.

### Story + Context + Real-World Relevance

Alexandra understands that for any new quantitative strategy, a comprehensive comparison against established benchmarks is essential. This helps in understanding the true value-add of the RL approach and identifying scenarios where it outperforms or underperforms. She will implement:

1.  **Equal-Weight (EW):** A simple strategy where all assets are equally weighted and rebalanced daily. This represents a naive diversification.
2.  **Fixed 60/40 Stock-Bond:** A classic strategic asset allocation, rebalanced monthly. This is a common institutional benchmark. For this portfolio, Alexandra defines the fixed weights as 60% SPY, 30% TLT, 5% GLD, and 5% SHY.
3.  **Static Markowitz Optimization:** Weights are optimized once at the beginning of the test period using historical data (e.g., the training period) to maximize the Sharpe ratio (or minimize variance for a target return) and remain fixed thereafter. This is the cornerstone of modern portfolio theory, but it's static. For simplicity and to show a static allocation, we'll compute minimum variance weights based on the training data.
4.  **Monthly Rebalanced 60/40:** A tactical strategy that rebalances back to a 60/40 target (60% SPY, 30% TLT, 5% GLD, 5% SHY) at the start of each month, allowing for drift within the month. This introduces periodic rebalancing without dynamic adaptation to market regimes.

```python
def run_baseline_strategy(returns_df, weights_fn, name, initial_value, tc_rate=0.001):
    """
    Runs a baseline portfolio strategy over the given returns data.
    
    Args:
        returns_df (pd.DataFrame): Daily returns for assets.
        weights_fn (callable): Function that returns asset weights (t, df, prev_weights).
        name (str): Name of the strategy.
        initial_value (float): Starting portfolio value.
        tc_rate (float): Transaction cost rate per unit of turnover.
        
    Returns:
        tuple: (list, list) of portfolio values and historical weights.
    """
    values = [initial_value]
    weights_hist = []
    prev_weights = np.ones(returns_df.shape[1]) / returns_df.shape[1] # Start with equal weights

    for t in range(len(returns_df)):
        # Determine new weights based on strategy
        w = weights_fn(t, returns_df.iloc[:t+1], prev_weights)

        # Calculate turnover and transaction costs
        turnover = np.sum(np.abs(w - prev_weights))
        tc = tc_rate * turnover * values[-1]

        # Calculate portfolio return
        port_ret = np.dot(w, returns_df.iloc[t].values)
        
        # Update portfolio value
        current_value = values[-1] * (1 + port_ret) - tc
        values.append(current_value)
        weights_hist.append(w)
        prev_weights = w # Update previous weights for next iteration

    print(f"Strategy '{name}' simulated.")
    return values, weights_hist


# --- Baseline Weight Functions ---

# 1. Equal-Weight (rebalanced daily)
def equal_weight_fn(t, df, pw):
    """Returns equal weights for all assets."""
    return np.ones(df.shape[1]) / df.shape[1]

# 2. Fixed 60/40 Stock-Bond (rebalanced monthly)
FIXED_60_40_WEIGHTS = np.array([0.60, 0.30, 0.05, 0.05]) # SPY, TLT, GLD, SHY
def monthly_rebal_60_40_fn(t, df, pw):
    """Rebalances to 60/40 at the start of each month, otherwise holds."""
    if t == 0 or (df.index[t].month != df.index[t-1].month):
        return FIXED_60_40_WEIGHTS
    return pw # Keep previous weights for the rest of the month (allow drift)

# 3. Static Markowitz Optimization (Min-Variance, computed once on training data)
from sklearn.covariance import LedoitWolf

# Compute covariance and mean from training data
train_cov_annualized = LedoitWolf().fit(train_returns).covariance_ * 252 # Annualize for consistency
train_mu_annualized = train_returns.mean().values * 252

# Minimum variance weights (long-only constraint)
# min w'Sigma w subject to sum(w) = 1, w_i >= 0
# For minimum variance, we can derive the analytical solution for unconstrained,
# then project to constrained if needed. A simpler approach is to use inverse covariance.
# More generally, for long-only, it's a quadratic programming problem.
# For demonstration, we'll use a simplified inverse covariance approach for minimum variance
# without explicit w_i >= 0 for now, then project.
cov_inv = np.linalg.pinv(train_cov_annualized)
mv_weights_unnormalized = cov_inv @ np.ones(train_returns.shape[1])
mv_weights_static = mv_weights_unnormalized / np.sum(mv_weights_unnormalized)

# Apply long-only constraint and re-normalize if necessary
mv_weights_static[mv_weights_static < 0] = 0
mv_weights_static /= np.sum(mv_weights_static)

def static_markowitz_fn(t, df, pw):
    """Returns the pre-computed static Markowitz weights."""
    return mv_weights_static

# 4. Monthly Rebalanced Markowitz (based on current month's data, simplified for example)
# For a truly dynamic Markowitz, one would re-optimize monthly using *available* data.
# Here, for simplicity and to highlight the 'monthly rebalance' aspect, we will use
# a similar principle to the 60/40 monthly rebalance but with Markowitz weights.
# However, the static Markowitz computed above is more illustrative of a fixed target.
# Let's keep the baseline to the 60/40 monthly rebalance as that is easier to manage in this context.
# We will use the monthly rebalanced 60/40 as the fourth baseline, as described in the requirements.

# Baseline strategies dictionary
BASELINE_STRATEGIES = {
    'Equal-Weight': equal_weight_fn,
    'Fixed 60/40': lambda t, df, pw: FIXED_60_40_WEIGHTS, # Fixed, no rebalancing logic in fn
    'Static Markowitz': static_markowitz_fn,
    'Monthly Rebalanced 60/40': monthly_rebal_60_40_fn
}

# Dictionary to store results
baseline_results = {}

# Run baselines on the test data
print("Running baseline strategies on the test period (2020-01-01 to 2024-12-31)...")
for name, fn in BASELINE_STRATEGIES.items():
    # Note: For 'Fixed 60/40', the `fn` itself returns constant weights,
    # so the `run_baseline_strategy` function handles daily turnover if the fn provides fixed weights,
    # but the `monthly_rebal_60_40_fn` explicitly handles the monthly rebalance condition.
    # For 'Fixed 60/40', we intend it to be buy-and-hold once set, but rebalance if drift is too much.
    # The monthly_rebal_60_40_fn does this better. Let's make the 'Fixed 60/40' also monthly rebalanced for a fair comparison of "static target" vs "dynamic target".
    
    # Re-adjusting "Fixed 60/40" to use monthly rebalancing for a more realistic baseline
    # The 'Fixed 60/40' name implies a target that is maintained.
    if name == 'Fixed 60/40':
        values, weights_hist = run_baseline_strategy(test_returns, monthly_rebal_60_40_fn, name, INITIAL_PORTFOLIO_VALUE, TRANSACTION_COST_RATE)
    else:
        values, weights_hist = run_baseline_strategy(test_returns, fn, name, INITIAL_PORTFOLIO_VALUE, TRANSACTION_COST_RATE)
    baseline_results[name] = {'values': values, 'weights': weights_hist}

print("\nAll baseline strategies simulated.")
```

### Explanation of Execution

Alexandra has successfully simulated four common portfolio strategies over the test period. Each strategy has a distinct rebalancing approach and risk profile. The `run_baseline_strategy` function meticulously tracks portfolio value, asset weights, and transaction costs for each, ensuring a consistent and fair comparison framework. These results will serve as crucial benchmarks for evaluating the RL agent's performance.

---

## 6. Evaluating the RL Agent and Baselines on Out-of-Sample Data

Now, Alexandra will evaluate the trained RL agent on the unseen test dataset and compare its performance metrics (Sharpe Ratio, Max Drawdown, Total Return, Turnover) against the established baseline strategies. This is the ultimate test of the RL agent's adaptiveness and robustness in real-world market conditions.

### Story + Context + Real-World Relevance

For Alexandra, performance evaluation is not just about raw returns. Risk-adjusted returns (Sharpe Ratio), downside risk (Max Drawdown), and trading efficiency (Turnover) are equally, if not more, important. High turnover can erode returns through transaction costs, making a strategy impractical. The test period (2020-2024) is deliberately chosen to include significant market shocks (COVID-19, 2022 rate hikes), providing a rigorous test of the agent's ability to navigate volatile regimes.

$$ \text{Sharpe Ratio} = \frac{\text{Mean(Portfolio Returns) - Risk-Free Rate}}{\text{Std Dev(Portfolio Returns)}} \times \sqrt{\text{Annualization Factor}} $$
Here, we simplify by assuming a 0% risk-free rate, and the annualization factor is $\sqrt{252}$ for daily data.

$$ \text{Max Drawdown} = \max_{t} \left( 1 - \frac{V_t}{\max_{s \le t} V_s} \right) $$
where $V_t$ is the portfolio value at time $t$, and $\max_{s \le t} V_s$ is the peak portfolio value up to time $t$.

$$ \text{Turnover} = \frac{1}{T} \sum_{t=1}^{T} ||\mathbf{w}_t - \mathbf{w}_{t-1}||_1 $$
where $\mathbf{w}_t$ is the weight vector at time $t$, and $T$ is the number of rebalancing periods. For daily rebalancing, this is the average daily L1 norm of weight changes.

```python
def evaluate_strategy_performance(portfolio_values, initial_value, weights_history, tc_rate=TRANSACTION_COST_RATE):
    """
    Calculates key performance metrics for a given strategy.
    
    Args:
        portfolio_values (list): List of portfolio values over time.
        initial_value (float): Initial portfolio value.
        weights_history (list): List of weight vectors over time.
        tc_rate (float): Transaction cost rate (used for turnover calculation if not implicitly handled).
        
    Returns:
        dict: Dictionary of performance metrics.
    """
    # Convert to numpy array for calculations
    values = np.array(portfolio_values)
    
    # Calculate daily returns
    daily_rets = np.diff(values) / values[:-1]
    
    # Total Return
    total_return = (values[-1] / initial_value) - 1
    
    # Annualized Sharpe Ratio (assuming 0% risk-free rate)
    if np.std(daily_rets) != 0:
        sharpe_ratio = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0 # Handle cases with zero volatility

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(values)
    drawdowns = (cumulative_max - values) / cumulative_max
    max_drawdown = np.max(drawdowns)

    # Turnover (average absolute change in weights per step)
    total_turnover = 0.0
    if len(weights_history) > 1:
        weights_array = np.array(weights_history)
        # Turnover is sum of absolute changes in weights for each asset, averaged daily
        daily_turnover = np.sum(np.abs(weights_array[1:] - weights_array[:-1]), axis=1)
        total_turnover = np.mean(daily_turnover) * 252 # Annualize average daily turnover
    else:
        total_turnover = 0.0 # No turnover if only one allocation

    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Annualized Turnover': total_turnover
    }

# --- Evaluate RL Agent ---
print("--- Evaluating RL Agent ---")
# Reset the test environment
test_env_instance = PortfolioAllocEnv(
    returns_df=test_returns, 
    features_df=test_features, 
    initial_value=INITIAL_PORTFOLIO_VALUE, 
    tc_rate=TRANSACTION_COST_RATE, 
    risk_aversion=RISK_AVERSION_COEFF,
    reward_type=REWARD_FUNCTION_TYPE
)
obs, info = test_env_instance.reset()

rl_values = [info['portfolio_value']]
rl_weights_hist = [info['weights']]

done = False
while not done:
    action, _ = ppo_model.predict(obs, deterministic=True) # deterministic=True for evaluation
    obs, reward, done, truncated, info = test_env_instance.step(action)
    rl_values.append(info['portfolio_value'])
    rl_weights_hist.append(info['weights'])

# Remove the initial equal weight from rl_weights_hist as it's not an action of the agent
rl_weights_hist = rl_weights_hist[1:] 

rl_metrics = evaluate_strategy_performance(rl_values, INITIAL_PORTFOLIO_VALUE, rl_weights_hist)
print(f"{'RL Agent (PPO)':30s}: Return={rl_metrics['Total Return']:.2%}, "
      f"Sharpe={rl_metrics['Sharpe Ratio']:.3f}, MaxDD={rl_metrics['Max Drawdown']:.1%}, "
      f"Turnover={rl_metrics['Annualized Turnover']:.2f}")

# Store RL results for comparison
all_strategy_metrics = {'RL Agent (PPO)': rl_metrics}
strategy_equity_curves = {'RL Agent (PPO)': rl_values}


# --- Evaluate Baseline Strategies ---
print("\n--- Evaluating Baseline Strategies ---")
for name, results in baseline_results.items():
    # Note: `weights_hist` from `run_baseline_strategy` does not include the initial equal weights setup,
    # it starts from the first rebalance. So `weights_hist` length is `len(returns_df)`.
    # `portfolio_values` includes initial value, so `len(portfolio_values)` is `len(returns_df) + 1`.
    
    # For turnover calculation, if weights_hist is empty (e.g. for a trivial strategy on single day), handle it.
    # Otherwise, ensure it aligns with the values to calculate daily turnover correctly.
    metrics = evaluate_strategy_performance(results['values'], INITIAL_PORTFOLIO_VALUE, results['weights'])
    
    all_strategy_metrics[name] = metrics
    strategy_equity_curves[name] = results['values']
    print(f"{name:30s}: Return={metrics['Total Return']:.2%}, "
          f"Sharpe={metrics['Sharpe Ratio']:.3f}, MaxDD={metrics['Max Drawdown']:.1%}, "
          f"Turnover={metrics['Annualized Turnover']:.2f}")

```

### Explanation of Execution

Alexandra now has a clear quantitative overview of how the RL agent performed against traditional strategies during a challenging out-of-sample period. She can see the total return, risk-adjusted return (Sharpe), worst-case scenario loss (Max Drawdown), and the cost efficiency (Turnover) for each. This allows her to determine if the adaptive RL policy truly delivered a competitive edge, especially considering the impact of transaction costs on turnover.

---

## 7. Visualizing Policy Behavior and Performance Insights

To gain deeper insights, Alexandra will visualize the performance and the RL agent's learned allocation policy. This helps in understanding *when* and *how* the agent adapts to different market conditions, which is crucial for building trust and explainability for stakeholders.

### Story + Context + Real-World Relevance

Visualizations are essential for Alexandra to communicate the strategy's efficacy and interpret its behavior.
*   **Equity Curves:** Show the cumulative wealth growth of each strategy, providing a high-level performance comparison.
*   **Stacked Weight Allocation (RL Agent):** This heatmap-like visualization (stacked area chart) reveals the dynamic shifts in the RL agent's portfolio weights over time. Alexandra can correlate these shifts with market events (e.g., increased gold allocation during volatility spikes, decreased equities during downturns), demonstrating the "regime-adaptive" nature of the policy. This is the core "policy interpretation" tool.
*   **Market Volatility Context:** Plotting rolling market volatility alongside the allocation shifts provides context, allowing Alexandra to see if the agent's rebalancing aligns with risk-on/risk-off indicators.
*   **Performance Comparison Bar Chart:** A direct visual comparison of key metrics (Sharpe, Max Drawdown, Total Return, Turnover) helps quickly identify strengths and weaknesses.

```python
# Convert performance metrics to a DataFrame for easier plotting
metrics_df = pd.DataFrame(all_strategy_metrics).T

# --- Visualization 1: Three-Panel Dashboard (Equity Curves, RL Weights, Market Volatility) ---
print("\nGenerating Three-Panel Dashboard...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})

# Panel 1: Equity Curves
for name, values in strategy_equity_curves.items():
    ax1.plot(test_returns.index[:len(values)], values, label=f'{name} (Sharpe={all_strategy_metrics[name]["Sharpe Ratio"]:.2f})')
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.set_title('Dynamic Asset Allocation: RL Agent vs. Baselines - Equity Curves', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='x', rotation=45)


# Panel 2: Stacked Area Chart of RL Agent's Asset Weight Allocations
rl_weights_df = pd.DataFrame(rl_weights_hist, columns=ASSET_TICKERS.values(), index=test_returns.index[:len(rl_weights_hist)])
colors = ['#2196F3', '#FF9800', '#FFD700', '#9E9E9E'] # Blue (SPY), Orange (TLT), Gold (GLD), Grey (SHY)
ax2.stackplot(rl_weights_df.index, rl_weights_df.T, labels=rl_weights_df.columns, colors=colors, alpha=0.8)
ax2.set_ylabel('Portfolio Weight', fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_title('RL Agent: Learned Allocation Over Time', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(axis='x', rotation=45)


# Panel 3: Rolling Volatility for Market Context
# Use SPY volatility as a proxy for overall market volatility
rolling_vol = test_returns['Equities'].rolling(21).std() * np.sqrt(252)
ax3.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.3, color='red')
ax3.set_ylabel('Equity Volatility (Ann.)', fontsize=12)
ax3.set_title('Market Context: Equity Realized Volatility', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('rl_dynamic_allocation_dashboard.png', dpi=150)
plt.show()
print("Three-Panel Dashboard saved as 'rl_dynamic_allocation_dashboard.png'")


# --- Visualization 2: Bar Chart Comparing Performance Metrics ---
print("\nGenerating Performance Comparison Bar Chart...")
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=False) # Separate Y-axes for different metrics
metrics_to_plot = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Annualized Turnover']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'mediumpurple', 'gold', 'teal'] # More colors than strategies

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    metric_values = metrics_df[metric]
    metric_values.plot(kind='bar', ax=ax, color=colors[:len(metric_values)], legend=False)
    ax.set_title(metric, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if 'Return' in metric or 'Drawdown' in metric: # Format as percentage
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    elif 'Turnover' in metric: # Format as percentage
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0f}%'.format(x) for x in vals]) # Integer percent
    else:
        ax.tick_params(axis='y')

plt.tight_layout()
plt.savefig('performance_comparison_bar_chart.png', dpi=150)
plt.show()
print("Performance Comparison Bar Chart saved as 'performance_comparison_bar_chart.png'")
```

### Explanation of Execution

Alexandra reviews the visualizations. The three-panel dashboard immediately shows the RL agent's dynamic behavior. She can see how the stacked area chart of asset weights shifts in response to the market volatility shown in the bottom panel. For example, during high-volatility periods like March 2020 (COVID-19 crash), she would expect to see equity (SPY) weights decrease and safe-haven assets (GLD, SHY) increase. The bar chart provides a concise summary of the quantitative results, allowing her to easily compare the Sharpe Ratio, Max Drawdown, Total Return, and critically, the Turnover for each strategy. This helps her in the 'policy interpretation' process, confirming if the agent learned financially sensible behaviors and if its adaptive nature translated into superior risk-adjusted returns and efficient trading.

---

## 8. Analyzing Regime-Conditional Allocation and Transaction Cost Efficiency

Alexandra concludes her analysis by examining the RL agent's behavior during specific market regimes and assessing the impact of transaction costs. This critical step provides deeper insights into the agent's intelligence and its practical applicability.

### Story + Context + Real-World Relevance

A key advantage of dynamic allocation is its ability to adapt to changing market conditions. Alexandra wants to explicitly see if the RL agent exhibited "risk-on" or "risk-off" behavior, for instance, by reducing equity exposure and increasing safe-haven assets during high-volatility periods. This helps validate the agent's learning process.

Furthermore, transaction costs are a *reality check* for any high-frequency rebalancing strategy. If an agent performs well but incurs massive turnover, its net performance will be severely eroded in practice. Alexandra needs to understand the trade-off the RL agent makes between performance and trading costs.

```python
def analyze_regime_conditional_allocation(weights_df, features_df):
    """
    Analyzes RL agent's average allocation during high vs. low volatility regimes.
    
    Args:
        weights_df (pd.DataFrame): RL agent's historical weights.
        features_df (pd.DataFrame): Market features, including equity volatility.
        
    Returns:
        pd.DataFrame: Average weights for high-vol and low-vol regimes.
    """
    # Align indices
    aligned_features = features_df.loc[weights_df.index]
    
    # Use SPY volatility as a proxy for overall market volatility
    # Assuming 'Equities_vol_21d' is the SPY volatility feature
    equity_vol = aligned_features['Equities_vol_21d']
    
    # Define volatility threshold (e.g., top/bottom 30% of volatility)
    vol_threshold_high = equity_vol.quantile(0.70)
    vol_threshold_low = equity_vol.quantile(0.30)
    
    high_vol_days = equity_vol[equity_vol > vol_threshold_high].index
    low_vol_days = equity_vol[equity_vol < vol_threshold_low].index
    
    avg_weights_high_vol = weights_df.loc[high_vol_days].mean()
    avg_weights_low_vol = weights_df.loc[low_vol_days].mean()
    
    regime_weights = pd.DataFrame({
        'Low Volatility Regime': avg_weights_low_vol,
        'High Volatility Regime': avg_weights_high_vol
    })
    
    return regime_weights

print("--- Regime-Conditional Allocation Analysis ---")
regime_allocations = analyze_regime_conditional_allocation(rl_weights_df, test_features)
print("Average RL Agent Weights by Market Volatility Regime:")
print(regime_allocations.applymap(lambda x: f"{x:.2%}")) # Format as percentages


def compare_turnover_efficiency(all_strategy_metrics):
    """
    Compares the annualized turnover across all strategies.
    
    Args:
        all_strategy_metrics (dict): Dictionary containing metrics for all strategies.
        
    Returns:
        pd.DataFrame: Turnover metrics for all strategies.
    """
    turnover_data = {name: metrics['Annualized Turnover'] for name, metrics in all_strategy_metrics.items()}
    turnover_series = pd.Series(turnover_data)
    turnover_df = pd.DataFrame(turnover_series, columns=['Annualized Turnover'])
    return turnover_df.sort_values(by='Annualized Turnover', ascending=False)

print("\n--- Transaction Cost & Turnover Efficiency Analysis ---")
turnover_comparison = compare_turnover_efficiency(all_strategy_metrics)
print("Annualized Turnover Comparison (Higher is more trading):")
print(turnover_comparison.applymap(lambda x: f"{x:.2f}")) # Format as raw turnover value
```

### Explanation of Execution

Alexandra examines the average weights during low and high-volatility regimes. She looks for evidence of an intelligent policy: for instance, a decrease in equity exposure and an increase in bond or gold allocation during high-volatility periods would indicate a successful "risk-off" adaptation. This directly validates whether the RL agent truly learned regime-adaptive allocation.

The turnover comparison provides crucial financial context. If the RL agent's turnover is significantly higher than passive strategies, it means the perceived performance gains might be eroded by real-world trading costs. A lower turnover for comparable returns would be a strong indicator of the agent's efficiency and practical utility. This analysis helps Alexandra determine if the RL agent offers a realistic and implementable improvement to Alpha Capital's portfolio management strategies.
