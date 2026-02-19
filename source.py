import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration Constants ---
# These can be made configurable via function arguments in an app.py
ASSET_TICKERS = {'SPY': 'Equities', 'TLT': 'Long Bonds', 'GLD': 'Gold', 'SHY': 'Cash'}
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
TRAIN_END_DATE = '2019-12-31'

INITIAL_PORTFOLIO_VALUE = 100000
TRANSACTION_COST_RATE = 0.001  # 10 basis points per unit of turnover
RISK_AVERSION_COEFF = 2.0      # Lambda for Mean-Variance Utility
REWARD_FUNCTION_TYPE = 'mean_variance'  # Options: 'mean_variance', 'sortino', 'log_wealth'

TOTAL_TIMESTEPS = 5000       # Total timesteps for PPO training
MODEL_SAVE_PATH = "ppo_portfolio_agent.zip"
PLOT_FILENAME = 'rl_dynamic_allocation_dashboard.png'

# --- 1. Data Loading and Preparation Functions ---

def load_and_prepare_data(tickers_dict, start_date, end_date,
                          min_window=63, mom_window=5, vol_window=21):
    """
    Downloads historical Close prices, calculates daily returns,
    and engineers market features for the RL environment.

    Args:
        tickers_dict (dict): Dictionary mapping ticker symbols to friendly names.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).
        min_window (int): Minimum window for feature calculation (e.g., cross-correlation).
        mom_window (int): Window for momentum feature.
        vol_window (int): Window for volatility feature.

    Returns:
        tuple: (returns_df, features_df)
            returns_df (pd.DataFrame): Daily returns, columns=asset names, values=daily returns.
            features_df (pd.DataFrame): Engineered features aligned to returns_df index.
    """
    tickers = list(tickers_dict.keys())
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if raw is None or len(raw) == 0:
        print("Error: No data downloaded from yfinance. Check tickers or date range.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Robustly extract Close ---
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Close" not in raw.columns.get_level_values(0)):
            print("Error: 'Close' not found in downloaded data.")
            return pd.DataFrame(), pd.DataFrame()
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            print("Error: 'Close' not found in downloaded data.")
            return pd.DataFrame(), pd.DataFrame()
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]] # Rename for single ticker case

    close = close.dropna(how="all")
    if close.empty:
        print("Error: Close price frame empty after dropping all-NaN rows.")
        return pd.DataFrame(), pd.DataFrame()

    close = close.rename(columns={t: tickers_dict[t] for t in close.columns if t in tickers_dict})

    returns = close.pct_change().dropna(how="any")

    if returns.empty or returns.shape[1] == 0:
        print("Error: Returns DataFrame empty after pct_change().dropna().")
        return pd.DataFrame(), pd.DataFrame()

    if returns.shape[0] < min_window:
        print(
            f"Warning: Not enough return observations ({returns.shape[0]}) "
            f"for {min_window}-day rolling features. Returning returns but empty features."
        )
        return returns, pd.DataFrame()

    # --- Feature engineering ---
    feature_list = []

    # Per-asset momentum and volatility features
    for asset in returns.columns:
        mom = returns[asset].rolling(mom_window).mean().rename(f"{asset}_ret_{mom_window}d")
        vol = (returns[asset].rolling(vol_window).std() * np.sqrt(252)).rename(f"{asset}_vol_{vol_window}d")
        feature_list.extend([mom, vol])

    # Cross-correlation mean (scalar per day)
    cross_corr = pd.Series(index=returns.index, dtype=float, name=f"cross_corr_mean_{min_window}d")
    for i in range(min_window - 1, len(returns)):
        window = returns.iloc[i - min_window + 1 : i + 1]
        corr = window.corr()
        vals = corr.values
        n = vals.shape[0]
        if n <= 1:
            cross_corr.iloc[i] = np.nan
        else:
            off_diag = vals[~np.eye(n, dtype=bool)]
            cross_corr.iloc[i] = np.nanmean(off_diag)
    feature_list.append(cross_corr)

    features = pd.concat(feature_list, axis=1).dropna(how="any")

    if features.empty:
        print("Error: Features DataFrame empty after concat/dropna (likely too many NaNs).")
        return returns, pd.DataFrame()

    returns = returns.loc[features.index]

    print(f"\nReturns shape: {returns.shape} | Features shape: {features.shape}")
    print(f"Assets: {list(returns.columns)}")
    print(f"Features per step: {features.shape[1]}")
    print(f"Feature date range: {features.index[0].date()} → {features.index[-1].date()}")

    return returns, features

def split_data_into_train_test(all_returns, all_features, train_end_date):
    """
    Splits the data into training and testing sets based on the specified date.

    Args:
        all_returns (pd.DataFrame): DataFrame of all returns.
        all_features (pd.DataFrame): DataFrame of all features.
        train_end_date (str): End date for the training period (YYYY-MM-DD).

    Returns:
        tuple: (train_returns, train_features, test_returns, test_features)
    """
    if all_features.empty:
        print("\nSkipping train/test split as features DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    split_dt = pd.to_datetime(train_end_date)

    train_mask = all_features.index <= split_dt
    test_mask  = all_features.index > split_dt

    train_returns = all_returns.loc[train_mask]
    train_features = all_features.loc[train_mask]

    test_returns = all_returns.loc[test_mask]
    test_features = all_features.loc[test_mask]

    if train_returns.shape[1] == 0:
        raise ValueError("train_returns has 0 columns (no assets). Check ticker mapping / download output.")
    if train_returns.shape[0] < 2:
        raise ValueError("train_returns has too few rows after feature alignment/split.")

    print(f"\nTraining (returns): {train_returns.shape}, (features): {train_features.shape}")
    print(f"Testing  (returns): {test_returns.shape}, (features): {test_features.shape}")

    return train_returns, train_features, test_returns, test_features


# --- 2. Portfolio Allocation Environment Class ---

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
        self.initial_value = initial_value
        self.tc_rate = tc_rate
        self.risk_aversion = risk_aversion
        self.reward_type = reward_type
        self.n_steps = len(returns_df) - 1  # Number of trading days in environment

        # Continuous action space: raw weights (softmax-normalized)
        self.action_space = spaces.Box(
              low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation space: current weights + market features
        obs_dim = self.n_assets + features_df.shape[1]
        self.observation_space = spaces.Box(
              low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize environment state
        self.reset()

    def _softmax(self, x):
        """Converts raw action outputs to valid portfolio weights (sums to 1)."""
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()

    def _get_obs(self):
        """Returns the current observation for the agent."""
        # Current portfolio weights + market features for the current step index
        current_features = self.features[self.step_idx]
        return np.concatenate([
            self.weights,
            current_features
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
            if len(self.returns_history) < 21:  # Need at least 21 days for rolling variance
                  return port_return  # No variance penalty initially

            recent_returns = np.array(self.returns_history[-21:])
            portfolio_variance = np.var(recent_returns) * 252  # Annualize variance
            reward = port_return - self.risk_aversion * portfolio_variance
            return reward

        elif self.reward_type == 'sortino':
            current_value = self.portfolio_value
            peak_value = np.max(self.portfolio_value_history)
            drawdown = max(0, (peak_value - current_value) / peak_value)
            reward = port_return - self.risk_aversion * drawdown
            return reward

        elif self.reward_type == 'log_wealth':
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
        new_weights = self._softmax(action)

        # Calculate turnover and transaction costs
        turnover = np.sum(np.abs(new_weights - self.weights))
        tc = self.tc_rate * turnover * self.portfolio_value

        asset_returns = self.returns[self.step_idx]
        port_return = np.dot(new_weights, asset_returns)

        # Update portfolio value
        self.portfolio_value *= (1 + port_return)
        self.portfolio_value -= tc  # Deduct transaction costs

        # Update weights and history
        self.weights = new_weights
        self.returns_history.append(port_return)
        self.portfolio_value_history.append(self.portfolio_value)

        # Compute reward
        reward = self._compute_reward(port_return)

        # Advance step
        self.step_idx += 1
        done = self.step_idx >= self.n_steps
        truncated = False

        info = {
              'portfolio_value': self.portfolio_value,
            'weights': new_weights.copy(),
            'turnover': turnover,
            'port_return': port_return,
            'transaction_cost': tc
        }
        # Get obs for the next state, or final obs if done
        next_obs = self._get_obs()

        return next_obs, reward, done, truncated, info

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
        self.weights = np.ones(self.n_assets) / self.n_assets  # Start with equal weights
        self.returns_history = []
        self.portfolio_value_history = [self.initial_value]

        info = {
              'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy()
        }
        return self._get_obs(), info


# --- 3. Model Training Function ---

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
          "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log="./ppo_portfolio_tensorboard/"
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_save_path)
    print(f"PPO training complete. Model saved to {model_save_path}")
    return model


# --- 4. Baseline Strategy Functions ---

def run_baseline_strategy(returns_df, weights_fn, name, initial_value, tc_rate=TRANSACTION_COST_RATE):
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
    prev_weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]  # Start with equal weights

    for t in range(len(returns_df)):
        # Determine new weights based on strategy
        w = weights_fn(t, returns_df.iloc[:t+1], prev_weights) # pass df slice up to current day

        # Calculate turnover and transaction costs
        turnover = np.sum(np.abs(w - prev_weights))
        tc = tc_rate * turnover * values[-1]

        # Calculate portfolio return
        port_ret = np.dot(w, returns_df.iloc[t].values)

        # Update portfolio value
        current_value = values[-1] * (1 + port_ret) - tc
        values.append(current_value)
        weights_hist.append(w)
        prev_weights = w  # Update previous weights for next iteration

    print(f"Strategy '{name}' simulated.")
    return values, weights_hist


def get_static_markowitz_weights(train_returns):
    """
    Computes static Markowitz Minimum Variance weights based on training data.

    Args:
        train_returns (pd.DataFrame): Training period returns.

    Returns:
        np.ndarray: Array of static Markowitz weights.
    """
    train_cov_annualized = LedoitWolf().fit(train_returns).covariance_ * 252

    # Minimum variance weights (long-only constraint, sum to 1)
    # Using inverse covariance and projecting to long-only
    cov_inv = np.linalg.pinv(train_cov_annualized)
    mv_weights_unnormalized = cov_inv @ np.ones(train_returns.shape[1])
    mv_weights_static = mv_weights_unnormalized / np.sum(mv_weights_unnormalized)

    # Apply long-only constraint and re-normalize if necessary
    mv_weights_static[mv_weights_static < 0] = 0
    mv_weights_static /= np.sum(mv_weights_static)  # Re-normalize after clamping

    return mv_weights_static

# --- 5. Evaluation Functions ---

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
    values = np.array(portfolio_values)
    if len(values) < 2:  # Handle cases where there's not enough data for returns/sharpe/drawdown
        return {
            'Total Return': (values[-1] / initial_value) - 1 if len(values) > 0 else 0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0,
            'Annualized Turnover': 0.0
        }

    daily_rets = np.diff(values) / values[:-1]
    total_return = (values[-1] / initial_value) - 1

    sharpe_ratio = 0.0
    if np.std(daily_rets) != 0:
          sharpe_ratio = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)

    cumulative_max = np.maximum.accumulate(values)
    drawdowns = (cumulative_max - values) / cumulative_max
    max_drawdown = np.max(drawdowns)

    total_turnover = 0.0
    if len(weights_history) > 1:
        weights_array = np.array(weights_history)
        if weights_array.ndim == 1 and len(weights_history) > 0 and isinstance(weights_history[0], np.ndarray):
             weights_array = np.stack(weights_history) # Re-stack if it's a list of arrays flattened to 1D initially
        elif weights_array.ndim == 1 and weights_array.shape[0] > 0: # single asset case
             weights_array = weights_array.reshape(-1, 1)

        if weights_array.shape[0] > 1:
            daily_turnover = np.sum(np.abs(weights_array[1:] - weights_array[:-1]), axis=1)
            total_turnover = np.mean(daily_turnover) * 252
    return {
          'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Annualized Turnover': total_turnover
    }

def run_rl_evaluation(ppo_model, test_returns, test_features, asset_tickers,
                      initial_value, tc_rate, risk_aversion, reward_type):
    """
    Evaluates the trained RL agent on the test environment.

    Args:
        ppo_model (PPO): The trained PPO model.
        test_returns (pd.DataFrame): Test period returns.
        test_features (pd.DataFrame): Test period features.
        asset_tickers (dict): Dictionary mapping ticker symbols to friendly names.
        initial_value (float): Initial portfolio value.
        tc_rate (float): Transaction cost rate.
        risk_aversion (float): Risk aversion coefficient.
        reward_type (str): Type of reward function used.

    Returns:
        tuple: (rl_metrics, rl_values, rl_weights_df)
            rl_metrics (dict): Performance metrics for the RL agent.
            rl_values (list): Historical portfolio values for the RL agent.
            rl_weights_df (pd.DataFrame): Historical weights for the RL agent.
    """
    print("--- Evaluating RL Agent ---")
    test_env_instance = PortfolioAllocEnv(
          returns_df=test_returns,
        features_df=test_features,
        initial_value=initial_value,
        tc_rate=tc_rate,
        risk_aversion=risk_aversion,
        reward_type=reward_type
    )
    obs, info = test_env_instance.reset()

    rl_values = [info['portfolio_value']]
    rl_weights_hist = [info['weights']]

    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)  # deterministic=True for evaluation
        obs, reward, done, truncated, info = test_env_instance.step(action)
        rl_values.append(info['portfolio_value'])
        rl_weights_hist.append(info['weights'])

    # Remove the initial equal weight from rl_weights_hist as it's not an action of the agent
    rl_weights_hist = rl_weights_hist[1:]

    rl_metrics = evaluate_strategy_performance(rl_values, initial_value, rl_weights_hist)
    print(f"{'RL Agent (PPO)':30s}: Return={rl_metrics['Total Return']:.2%}, "
          f"Sharpe={rl_metrics['Sharpe Ratio']:.3f}, MaxDD={rl_metrics['Max Drawdown']:.1%}, "
          f"Turnover={rl_metrics['Annualized Turnover']:.2f}")

    # Align rl_weights_hist to test_idx safely for plotting and analysis
    weights_arr = np.asarray(rl_weights_hist)
    if weights_arr.ndim == 1 and len(rl_weights_hist) > 0 and isinstance(rl_weights_hist[0], np.ndarray):
        weights_arr = np.stack(rl_weights_hist)
    elif weights_arr.ndim == 1: # Handle single asset case if `weights_arr` is still 1D
        weights_arr = weights_arr.reshape(-1, len(asset_tickers))

    m = min(len(weights_arr), len(test_returns.index))
    rl_weights_df = pd.DataFrame(weights_arr[:m], columns=list(asset_tickers.values()), index=test_returns.index[:m])

    return rl_metrics, rl_values, rl_weights_df

def run_baseline_evaluations(test_returns, initial_value, tc_rate, train_returns, asset_tickers):
    """
    Runs and evaluates all defined baseline strategies.

    Args:
        test_returns (pd.DataFrame): Test period returns.
        initial_value (float): Initial portfolio value.
        tc_rate (float): Transaction cost rate.
        train_returns (pd.DataFrame): Training period returns (used for Markowitz).
        asset_tickers (dict): Dictionary mapping ticker symbols to friendly names.

    Returns:
        tuple: (baseline_results, all_strategy_metrics, strategy_equity_curves)
    """
    print("\n--- Evaluating Baseline Strategies ---")

    # 1. Equal-Weight
    def equal_weight_fn(t, df, pw):
        return np.ones(df.shape[1]) / df.shape[1]

    # 2. Fixed 60/40 Stock-Bond (rebalanced monthly for fair turnover comparison)
    fixed_60_40_target_weights = np.array([0.60, 0.30, 0.05, 0.05]) # SPY, TLT, GLD, SHY, ensure order matches ASSET_TICKERS values
    if len(fixed_60_40_target_weights) != len(asset_tickers):
         print(f"Warning: FIXED_60_40_TARGET_WEIGHTS length ({len(fixed_60_40_target_weights)}) does not match asset count ({len(asset_tickers)}). Using equal weights for 60/40 baseline.")
         fixed_60_40_target_weights = np.ones(len(asset_tickers)) / len(asset_tickers)

    def monthly_rebal_60_40_fn(t, df, pw):
        if t == 0 or (df.index[t].month != df.index[t-1].month):
            return fixed_60_40_target_weights
        return pw # Hold previous weights for the rest of the month

    # 3. Static Markowitz Optimization (computed once on training data)
    static_markowitz_weights = get_static_markowitz_weights(train_returns)
    def static_markowitz_fn(t, df, pw):
        return static_markowitz_weights

    baseline_strategies_fns = {
        'Equal-Weight': equal_weight_fn,
        'Fixed 60/40': monthly_rebal_60_40_fn,  # Using monthly rebalance logic for fair comparison
        'Static Markowitz': static_markowitz_fn,
        'Monthly Rebalanced 60/40': monthly_rebal_60_40_fn # Same as Fixed 60/40 for consistency
    }

    baseline_results = {}
    all_strategy_metrics = {}
    strategy_equity_curves = {}

    for name, fn in baseline_strategies_fns.items():
        values, weights_hist = run_baseline_strategy(test_returns, fn, name, initial_value, tc_rate)
        metrics = evaluate_strategy_performance(values, initial_value, weights_hist)
        baseline_results[name] = {'values': values, 'weights': weights_hist}
        all_strategy_metrics[name] = metrics
        strategy_equity_curves[name] = values
        print(f"{name:30s}: Return={metrics['Total Return']:.2%}, "
              f"Sharpe={metrics['Sharpe Ratio']:.3f}, MaxDD={metrics['Max Drawdown']:.1%}, "
              f"Turnover={metrics['Annualized Turnover']:.2f}")

    return baseline_results, all_strategy_metrics, strategy_equity_curves


# --- 6. Plotting Function ---

def _index_for_equity_curve(values_len, test_index, day_before):
    """
    Helper function to build a datetime index that matches the equity curve length.
    Ensures that the index properly aligns with the number of data points.
    """
    n = values_len
    if n == len(test_index) + 1:
        return pd.Index([day_before]).append(test_index)
    if n == len(test_index):
        return test_index
    # Fallback: take the last n dates from the "with-initial" index
    full_index_potential = pd.Index([day_before]).append(test_index)
    if n <= len(full_index_potential):
        return full_index_potential[-n:]
    # If values are longer than available dates (rare), pad forward with business days
    extra = n - len(full_index_potential)
    future_dates = pd.bdate_range(full_index_potential[-1] + pd.Timedelta(days=1), periods=extra)
    return full_index_potential.append(future_dates)


def plot_dashboard(strategy_equity_curves, all_strategy_metrics, rl_weights_df, test_returns,
                   asset_tickers, plot_filename='rl_dynamic_allocation_dashboard.png'):
    """
    Generates a three-panel dashboard comparing strategies and RL agent's behavior.

    Args:
        strategy_equity_curves (dict): Dictionary of portfolio values for each strategy.
        all_strategy_metrics (dict): Dictionary of performance metrics for each strategy.
        rl_weights_df (pd.DataFrame): RL agent's historical weights.
        test_returns (pd.DataFrame): Test period returns (used for market context).
        asset_tickers (dict): Dictionary mapping ticker symbols to friendly names.
        plot_filename (str): Filename to save the plot.
    """
    print("\nGenerating Three-Panel Dashboard...")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 18),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 2, 1]}
    )

    test_idx = test_returns.index
    day_before_test_start = test_idx[0] - pd.Timedelta(days=1)

    # -----------------------
    # Panel 1: Equity Curves
    # -----------------------
    for name, values in strategy_equity_curves.items():
        values_arr = np.asarray(values).ravel()
        x_index = _index_for_equity_curve(len(values_arr), test_idx, day_before_test_start)

        # Ensure x_index and values_arr have matching lengths
        m = min(len(x_index), len(values_arr))
        if len(x_index) != len(values_arr):
            x_index = x_index[:m]
            values_arr = values_arr[:m]

        ax1.plot(
            x_index, values_arr,
            label=f'{name} (Sharpe={all_strategy_metrics[name]["Sharpe Ratio"]:.2f})'
        )

    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title('Dynamic Asset Allocation: RL Agent vs. Baselines - Equity Curves', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(axis='x', rotation=45)

    # --------------------------------------------
    # Panel 2: RL Agent Weights
    # --------------------------------------------
    # Define colors based on asset_tickers for consistency
    asset_names = list(asset_tickers.values())
    default_colors = ['#2196F3', '#FF9800', '#FFD700', '#9E9E9E', '#4CAF50', '#8BC34A']
    plot_colors = default_colors[:len(asset_names)]

    ax2.stackplot(
        rl_weights_df.index,
        rl_weights_df.T.values,
        labels=rl_weights_df.columns,
        colors=plot_colors,
        alpha=0.8
    )
    ax2.set_ylabel('Portfolio Weight', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_title('RL Agent: Learned Allocation Over Time', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(axis='x', rotation=45)

    # ----------------------------------------
    # Panel 3: Rolling Volatility (aligned)
    # ----------------------------------------
    # Use 'Equities' (SPY) for rolling volatility if available
    if 'Equities' in test_returns.columns:
        rolling_vol = test_returns['Equities'].rolling(21).std() * np.sqrt(252)
        ax3.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.3, color='red')
        ax3.set_ylabel('Equity Volatility (Ann.)', fontsize=12)
        ax3.set_title('Market Context: Equity Realized Volatility', fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.set_visible(False)
        print("Warning: 'Equities' not found in test_returns for volatility plot. Hiding Panel 3.")

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.show()
    print(f"Three-Panel Dashboard saved as '{plot_filename}'")


# --- 7. Advanced Analysis Functions ---

def analyze_regime_conditional_allocation(weights_df, features_df):
    """
    Analyzes RL agent's average allocation during high vs. low volatility regimes.

    Args:
          weights_df (pd.DataFrame): RL agent's historical weights.
        features_df (pd.DataFrame): Market features, including equity volatility.

    Returns:
          pd.DataFrame: Average weights for high-vol and low-vol regimes.
    """
    aligned_features = features_df.loc[weights_df.index]

    vol_feature_name = 'Equities_vol_21d'
    if vol_feature_name not in aligned_features.columns:
        print(f"Warning: Volatility feature '{vol_feature_name}' not found. Skipping regime analysis.")
        return pd.DataFrame()

    equity_vol = aligned_features[vol_feature_name]

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


# --- 8. Main Orchestration Function ---

def run_rl_portfolio_analysis(
    asset_tickers=ASSET_TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    train_end_date=TRAIN_END_DATE,
    initial_portfolio_value=INITIAL_PORTFOLIO_VALUE,
    transaction_cost_rate=TRANSACTION_COST_RATE,
    risk_aversion_coeff=RISK_AVERSION_COEFF,
    reward_function_type=REWARD_FUNCTION_TYPE,
    total_timesteps=TOTAL_TIMESTEPS,
    model_save_path=MODEL_SAVE_PATH,
    plot_filename=PLOT_FILENAME
):
    """
    Orchestrates the entire RL portfolio analysis workflow.

    Args:
        asset_tickers (dict): Mapping of ticker symbols to friendly names.
        start_date (str): Start date for data (YYYY-MM-DD).
        end_date (str): End date for data (YYYY-MM-DD).
        train_end_date (str): End date for training period (YYYY-MM-DD).
        initial_portfolio_value (float): Starting portfolio value.
        transaction_cost_rate (float): Transaction cost rate per unit of turnover.
        risk_aversion_coeff (float): Risk aversion coefficient for reward calculation.
        reward_function_type (str): Type of reward function ('mean_variance', 'sortino', 'log_wealth').
        total_timesteps (int): Total timesteps for PPO training.
        model_save_path (str): Path to save/load the trained PPO model.
        plot_filename (str): Filename to save the performance dashboard.

    Returns:
        tuple: (all_strategy_metrics, strategy_equity_curves, rl_weights_df, regime_allocations, turnover_comparison)
    """
    print("Starting RL Portfolio Analysis...")

    # 1. Load and Prepare Data
    all_returns, all_features = load_and_prepare_data(asset_tickers, start_date, end_date)

    if all_features.empty:
        print("Exiting: Data preparation failed or resulted in empty features.")
        return None, None, None, None, None

    train_returns, train_features, test_returns, test_features = split_data_into_train_test(
        all_returns, all_features, train_end_date
    )

    if train_features.empty or test_features.empty:
        print("Exiting: Train or test data split resulted in empty dataframes.")
        return None, None, None, None, None

    # 2. Create and Train RL Agent
    train_env_instance = PortfolioAllocEnv(
        returns_df=train_returns,
        features_df=train_features,
        initial_value=initial_portfolio_value,
        tc_rate=transaction_cost_rate,
        risk_aversion=risk_aversion_coeff,
        reward_type=reward_function_type
    )
    vec_env_train = DummyVecEnv([lambda: train_env_instance])
    print(f"Portfolio Allocation Environment created for training with {reward_function_type} reward and lambda={risk_aversion_coeff}.")
    print(f"Observation Space Shape: {vec_env_train.observation_space.shape}")
    print(f"Action Space Shape: {vec_env_train.action_space.shape}")

    ppo_model = train_ppo_agent(vec_env_train, total_timesteps, model_save_path)

    # 3. Evaluate RL Agent
    rl_metrics, rl_values, rl_weights_df = run_rl_evaluation(
        ppo_model, test_returns, test_features, asset_tickers,
        initial_portfolio_value, transaction_cost_rate, risk_aversion_coeff, reward_function_type
    )

    # 4. Run and Evaluate Baseline Strategies
    baseline_results, baseline_metrics, baseline_equity_curves = run_baseline_evaluations(
        test_returns, initial_portfolio_value, transaction_cost_rate, train_returns, asset_tickers
    )

    # Combine all results for plotting and final metric comparison
    all_strategy_metrics = {'RL Agent (PPO)': rl_metrics}
    all_strategy_metrics.update(baseline_metrics)

    strategy_equity_curves = {'RL Agent (PPO)': rl_values}
    strategy_equity_curves.update(baseline_equity_curves)

    # 5. Generate Dashboard
    plot_dashboard(strategy_equity_curves, all_strategy_metrics, rl_weights_df, test_returns, asset_tickers, plot_filename)

    # 6. Advanced Analysis
    print("\n--- Regime-Conditional Allocation Analysis ---")
    regime_allocations = analyze_regime_conditional_allocation(rl_weights_df, test_features)
    if not regime_allocations.empty:
        print("Average RL Agent Weights by Market Volatility Regime:")
        print(regime_allocations.applymap(lambda x: f"{x:.2%}"))
    else:
        print("Regime-conditional analysis skipped due to missing volatility features.")

    print("\n--- Transaction Cost & Turnover Efficiency Analysis ---")
    turnover_comparison = compare_turnover_efficiency(all_strategy_metrics)
    print("Annualized Turnover Comparison (Higher is more trading):")
    print(turnover_comparison.applymap(lambda x: f"{x:.2f}"))

    print("\nRL Portfolio Analysis Complete.")
    return all_strategy_metrics, strategy_equity_curves, rl_weights_df, regime_allocations, turnover_comparison


# --- Main Execution Block for Script ---
if __name__ == "__main__":
    print("All required libraries imported successfully.")

    # Call the main orchestration function with default or custom parameters
    final_metrics, final_equity_curves, final_rl_weights_df, final_regime_allocs, final_turnover_comps = \
        run_rl_portfolio_analysis(
            asset_tickers=ASSET_TICKERS,
            start_date=START_DATE,
            end_date=END_DATE,
            train_end_date=TRAIN_END_DATE,
            initial_portfolio_value=INITIAL_PORTFOLIO_VALUE,
            transaction_cost_rate=TRANSACTION_COST_RATE,
            risk_aversion_coeff=RISK_AVERSION_COEFF,
            reward_function_type=REWARD_FUNCTION_TYPE,
            total_timesteps=TOTAL_TIMESTEPS,
            model_save_path=MODEL_SAVE_PATH,
            plot_filename=PLOT_FILENAME
        )

    # Example of how you might access results after execution:
    if final_metrics:
        print("\n--- Summary Performance Metrics (DataFrame) ---")
        print(pd.DataFrame(final_metrics).T)
