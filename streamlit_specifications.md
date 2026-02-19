
# Streamlit Application Specification: Dynamic Asset Allocation via Reinforcement Learning

## 1. Application Overview

### Purpose
This Streamlit application serves as a hands-on lab for CFA Charterholders and Investment Professionals, like Alexandra Chen, to explore and implement dynamic asset allocation strategies using Reinforcement Learning (RL). It demonstrates how an adaptive, AI-driven policy can rebalance a multi-asset portfolio in response to changing market conditions, compare its performance against traditional benchmarks, and realistically account for transaction costs and risk preferences. The app aims to bridge classical portfolio management theory with modern RL practices, providing a framework for developing more resilient and adaptive portfolio management tools.

### High-Level Story Flow

Alexandra Chen, a Senior Portfolio Manager at Alpha Capital Management, will use this application to build and evaluate a PPO-based RL agent for dynamic asset allocation.

1.  **Introduction**: Alexandra starts with an overview of the case study, understanding the problem of non-stationary markets and the potential of RL.
2.  **Setup & Data Acquisition**: She defines her multi-asset universe, specifies historical data ranges, and triggers the data download and feature engineering process. This prepares the market "observations" for the RL agent.
3.  **Environment Design**: Alexandra configures the `gymnasium` environment, defining the rules of interaction for the RL agent. This includes selecting a risk-adjusted reward function (e.g., Mean-Variance Utility) and setting parameters like transaction costs and her risk aversion coefficient ($\lambda$).
4.  **Train RL Agent**: She initiates the training of a Proximal Policy Optimization (PPO) agent, allowing it to learn an adaptive allocation policy over historical data.
5.  **Baseline Strategies**: While the RL agent trains, she reviews and then executes simulations of four traditional benchmark strategies (Equal-Weight, Fixed 60/40, Static Markowitz, Monthly Rebalanced 60/40) over the same out-of-sample period.
6.  **Evaluate Performance**: After training and baseline simulations are complete, Alexandra evaluates all strategies on unseen test data, comparing key performance metrics like Sharpe ratio, Max Drawdown, Total Return, and Annualized Turnover.
7.  **Policy Insights**: Finally, she visualizes the RL agent's learned policy, examining how asset weights dynamically shift over time in response to market volatility, and performs a regime-conditional allocation analysis to understand the agent's "risk-on" / "risk-off" behavior and its transaction cost efficiency.

This structured workflow allows Alexandra to not only observe the RL agent's performance but also interpret its decision-making process, fostering trust and understanding in AI-driven investment strategies.

---

## 2. Code Requirements

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Assuming seaborn is implicitly used by matplotlib plotting functions from source.py

# Import all functions and constants from source.py
from source import *

# --- st.session_state Initialization ---
# Initialize session state variables if they don't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction"
if 'tickers_dict' not in st.session_state:
    st.session_state.tickers_dict = {'SPY': 'Equities', 'TLT': 'Long Bonds', 'GLD': 'Gold', 'SHY': 'Cash'}
if 'start_date' not in st.session_state:
    st.session_state.start_date = pd.to_datetime('2010-01-01')
if 'end_date' not in st.session_state:
    st.session_state.end_date = pd.to_datetime('2024-12-31')
if 'initial_portfolio_value' not in st.session_state:
    st.session_state.initial_portfolio_value = 100000
if 'transaction_cost_rate' not in st.session_state:
    st.session_state.transaction_cost_rate = 0.001
if 'risk_aversion_coeff' not in st.session_state:
    st.session_state.risk_aversion_coeff = 2.0
if 'reward_function_type' not in st.session_state:
    st.session_state.reward_function_type = 'mean_variance'
if 'total_timesteps' not in st.session_state:
    st.session_state.total_timesteps = 500000
if 'model_save_path' not in st.session_state:
    st.session_state.model_save_path = "ppo_portfolio_agent.zip"
if 'train_end_date' not in st.session_state:
    st.session_state.train_end_date = '2019-12-31' # Fixed as per requirements

# Data storage for results
if 'all_returns' not in st.session_state:
    st.session_state.all_returns = None
if 'all_features' not in st.session_state:
    st.session_state.all_features = None
if 'train_returns' not in st.session_state:
    st.session_state.train_returns = None
if 'train_features' not in st.session_state:
    st.session_state.train_features = None
if 'test_returns' not in st.session_state:
    st.session_state.test_returns = None
if 'test_features' not in st.session_state:
    st.session_state.test_features = None
if 'train_env_instance' not in st.session_state:
    st.session_state.train_env_instance = None
if 'vec_env_train' not in st.session_state:
    st.session_state.vec_env_train = None
if 'ppo_model' not in st.session_state:
    st.session_state.ppo_model = None
if 'rl_values' not in st.session_state:
    st.session_state.rl_values = None
if 'rl_weights_hist' not in st.session_state:
    st.session_state.rl_weights_hist = None
if 'baseline_results' not in st.session_state:
    st.session_state.baseline_results = None
if 'all_strategy_metrics' not in st.session_state:
    st.session_state.all_strategy_metrics = None
if 'strategy_equity_curves' not in st.session_state:
    st.session_state.strategy_equity_curves = None
if 'rl_weights_df' not in st.session_state:
    st.session_state.rl_weights_df = None
if 'regime_allocations' not in st.session_state:
    st.session_state.regime_allocations = None
if 'turnover_comparison' not in st.session_state:
    st.session_state.turnover_comparison = None

# --- Application Layout and Flow ---

st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    ["Introduction", "1. Setup & Data", "2. Environment Design", "3. Train RL Agent", "4. Baseline Strategies", "5. Evaluate Performance", "6. Policy Insights"]
)
st.session_state.current_page = page_selection

st.title("Dynamic Asset Allocation via Reinforcement Learning")

# --- Page: Introduction ---
if st.session_state.current_page == "Introduction":
    st.header("Case Study Introduction: Alpha Capital's Adaptive Edge")
    st.markdown(f"**Persona:** Alexandra Chen, CFA Charterholder and Senior Portfolio Manager at Alpha Capital Management.")
    st.markdown(f"**Organization:** Alpha Capital Management, an investment firm specializing in quantitative strategies for high-net-worth individuals and institutional clients.")
    st.markdown(f"""
Alexandra faces a persistent challenge: traditional static asset allocation models (like a fixed 60/40 stock-bond split or periodically rebalanced Markowitz portfolios) often underperform during rapidly changing market regimes. The financial landscape is increasingly non-stationary, with sudden shifts in volatility, correlations, and return profiles (e.g., the COVID-19 crash, 2022 rate hikes). Her goal is to explore how an adaptive, AI-driven policy can dynamically rebalance a multi-asset portfolio to achieve superior risk-adjusted returns while realistically accounting for transaction costs. She needs a robust framework that can learn to adjust asset weights in real-time, anticipating market shifts rather than reacting slowly.

This application will guide Alexandra through building a Reinforcement Learning (RL) agent capable of dynamic asset allocation. She will define a multi-asset investment environment, train a cutting-edge RL algorithm (PPO) to learn an optimal allocation policy, and rigorously compare its performance against several traditional benchmarks. The ultimate aim is to equip Alpha Capital with a more resilient and adaptive portfolio management tool.
""")
    st.markdown(f"---")
    st.header("Key Insight: Dynamic Markowitz with RL")
    st.markdown(f"""
This case bridges CFA portfolio management theory with RL practice. It applies RL concepts to the real portfolio management problem: choosing weights across multiple asset classes to maximize risk-adjusted returns over time. The key upgrade: **continuous actions** (weight vectors) instead of discrete (buy/sell/hold), requiring **policy gradient methods (PPO)** instead of Q-learning. This is the RL architecture actually used in production financial RL systems.

The comparison to Markowitz is deliberately pedagogical: participants trained on static MVO in the CFA curriculum now see how RL extends it to a dynamic, data-driven framework. The answer to "does RL beat Markowitz?" is nuanced: RL adapts faster but overfits more. The honest comparison—showing when each approach wins—is the case study's central insight.
""")

# --- Page: 1. Setup & Data ---
elif st.session_state.current_page == "1. Setup & Data":
    st.header("1. Setting Up the Dynamic Portfolio Lab")
    st.markdown(f"""
Alexandra begins by setting up her Python environment and defining her asset universe and data requirements. This section ensures her workspace is ready for data acquisition, RL environment creation, model training, and performance analysis.
""")

    st.subheader("Acquiring Market Data and Engineering Features")
    st.markdown(f"""
To build an intelligent allocation policy, Alexandra needs rich market data that captures the dynamics of her chosen asset classes. She will download historical prices for a diversified portfolio and then engineer relevant features that the RL agent can use to understand market conditions.

Alexandra's multi-asset portfolio includes:
*   **SPY:** S&P 500 ETF (Equities)
*   **TLT:** iShares 20+ Year Treasury Bond ETF (Long Bonds)
*   **GLD:** SPDR Gold Shares (Gold)
*   **SHY:** iShares 1-3 Year Treasury Bond ETF (Cash proxy / Short-Term Bonds)

These assets represent common building blocks in diversified portfolios. For her RL agent to make informed decisions, it needs to observe not just raw returns but also indicators of market state, such as short-term momentum, volatility, and cross-asset correlations. These features help the agent understand changing market regimes (e.g., risk-on vs. risk-off).

The dataset is split into a training period (2010-01-01 to 2019-12-31) for the RL agent to learn from, and a separate out-of-sample test period (2020-01-01 to 2024-12-31) for unbiased performance evaluation, which includes major market events like the COVID crash and the 2022 rate shock.
""")

    st.subheader("Configuration")
    tickers_options = ['SPY', 'TLT', 'GLD', 'SHY']
    selected_tickers = st.multiselect(
        "Select Asset Tickers:",
        options=tickers_options,
        default=list(st.session_state.tickers_dict.keys()),
        help="Select the asset classes for your portfolio."
    )
    # Update tickers_dict based on selection
    new_tickers_dict = {t: st.session_state.tickers_dict.get(t, t) for t in selected_tickers}
    st.session_state.tickers_dict = new_tickers_dict

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input(
            "Start Date:",
            value=st.session_state.start_date,
            min_value=pd.to_datetime('2000-01-01'),
            max_value=pd.to_datetime('2023-01-01'),
            help="Historical data start date."
        )
    with col2:
        st.session_state.end_date = st.date_input(
            "End Date:",
            value=st.session_state.end_date,
            min_value=pd.to_datetime('2020-01-01'),
            max_value=pd.to_datetime('2024-12-31'),
            help="Historical data end date. Test period runs until this date."
        )

    st.session_state.initial_portfolio_value = st.number_input(
        "Initial Portfolio Value:",
        min_value=10000.0,
        max_value=1000000.0,
        value=st.session_state.initial_portfolio_value,
        step=10000.0,
        help="Starting capital for the portfolio simulation."
    )
    st.session_state.transaction_cost_rate = st.number_input(
        "Transaction Cost Rate (e.g., 0.001 for 10 bps):",
        min_value=0.0,
        max_value=0.01,
        value=st.session_state.transaction_cost_rate,
        step=0.0001,
        format="%.4f",
        help="Cost per unit of turnover. E.g., 0.001 means 0.1% transaction cost."
    )

    if st.button("Load and Prepare Data"):
        if not st.session_state.tickers_dict:
            st.warning("Please select at least one asset ticker.")
        else:
            with st.spinner("Downloading and processing data... This may take a moment."):
                all_returns, all_features = load_and_prepare_data(
                    st.session_state.tickers_dict,
                    st.session_state.start_date.strftime('%Y-%m-%d'),
                    st.session_state.end_date.strftime('%Y-%m-%d')
                )
                
                # Define train and test masks based on features index for consistency
                train_mask = all_features.index <= pd.to_datetime(st.session_state.train_end_date)
                test_mask = all_features.index > pd.to_datetime(st.session_state.train_end_date)

                st.session_state.all_returns = all_returns
                st.session_state.all_features = all_features
                st.session_state.train_returns = all_returns[train_mask]
                st.session_state.train_features = all_features[train_mask]
                st.session_state.test_returns = all_returns[test_mask]
                st.session_state.test_features = all_features[test_mask]

            st.success("Data loaded and features engineered successfully!")
            st.write(f"**Training Data Shape (Returns):** {st.session_state.train_returns.shape}")
            st.write(f"**Training Data Shape (Features):** {st.session_state.train_features.shape}")
            st.write(f"**Testing Data Shape (Returns):** {st.session_state.test_returns.shape}")
            st.write(f"**Testing Data Shape (Features):** {st.session_state.test_features.shape}")
            st.write(f"**Assets:** {list(st.session_state.train_returns.columns)}")
            st.write(f"**Features per step:** {st.session_state.train_features.shape[1]}")
    else:
        if st.session_state.all_returns is not None:
            st.info("Data already loaded. Re-run to update with new parameters.")
            st.write(f"**Training Data Shape (Returns):** {st.session_state.train_returns.shape}")
            st.write(f"**Training Data Shape (Features):** {st.session_state.train_features.shape}")
            st.write(f"**Testing Data Shape (Returns):** {st.session_state.test_returns.shape}")
            st.write(f"**Testing Data Shape (Features):** {st.session_state.test_features.shape}")

# --- Page: 2. Environment Design ---
elif st.session_state.current_page == "2. Environment Design":
    st.header("2. Designing the Multi-Asset Portfolio Reinforcement Learning Environment")
    st.markdown(f"""
For the RL agent to learn, Alexandra must define the rules of its world: the portfolio environment. This involves specifying the state observations, the available actions (asset weights), the dynamics of portfolio value, and critically, how rewards are calculated based on risk-adjusted performance and transaction costs.
""")

    st.subheader("Key Components of the Environment Design")
    st.markdown(f"""
Alexandra needs to translate the complex world of portfolio management into a `gymnasium` environment. This is a standard practice in RL, allowing her to simulate market interactions. The agent will learn by interacting with this environment, receiving observations and rewards after each allocation decision.

Key components of her environment design include:
*   **State:** The agent needs to know its current portfolio allocation and the latest market features.
*   **Action Space:** The agent decides how to allocate capital across the assets. This must be a continuous action space (asset weights sum to 1), requiring a policy gradient method like PPO. The softmax function ensures valid portfolio weights from raw agent outputs:
""")
    st.markdown(r"$$ w_i = \frac{{e^{{a_i}}}}{{\sum_{{j=1}}^{{K}} e^{{a_j}}}} $$")
    st.markdown(r"where $w_i$ is the weight of asset $i$, $a_i$ is the raw action output for asset $i$, and $K$ is the number of assets. This ensures $w_i > 0$ and $\sum w_i = 1$.")
    st.markdown(f"""
*   **Portfolio Value Dynamics:** The portfolio value ($V_t$) evolves based on asset returns and is penalized by transaction costs ($TC_t$):
""")
    st.markdown(r"$$ V_{{t+1}} = V_t \cdot (1 + \mathbf{{w}}_t^{{\top}} \mathbf{{r}}_{{t+1}}) - TC_t $$")
    st.markdown(r"where $\mathbf{{w}}_t$ is the weight vector at time $t$, $\mathbf{{r}}_{{t+1}}$ is the vector of asset returns at $t+1$, and the transaction cost $TC_t$ is calculated based on turnover and a per-unit transaction rate:")
    st.markdown(r"$$ TC_t = c \cdot ||\mathbf{{w}}_{{t+1}} - \mathbf{{w}}_t||_1 \cdot V_t $$")
    st.markdown(r"Here, $c$ is the per-unit transaction cost rate (e.g., 0.001 for 10 bps), and $||\cdot||_1$ is the L1 norm, measuring total turnover.")
    st.markdown(f"""
*   **Reward Function (Utility-Based):** The reward function is crucial as it encodes Alexandra's investment policy statement (IPS) and risk preferences. It guides the agent to maximize risk-adjusted returns. She opts for the **Mean-Variance Utility** as the default, a direct connection to classical CFA utility theory ($U = E[r] - \frac{{1}}{{2}}A \sigma^2$):
""")
    st.markdown(r"$$ R_t = r_{{p,t}} - \lambda \sigma^2_{{p,t}} $$")
    st.markdown(r"where $r_{{p,t}}$ is the portfolio return at time $t$, $\sigma^2_{{p,t}}$ is the rolling 21-day annualized portfolio variance, and $\lambda$ is the risk aversion coefficient. A higher $\lambda$ indicates greater risk aversion, leading the agent to adopt more conservative strategies. This parameter is directly analogous to the risk aversion in a traditional IPS.")
    st.markdown(f"""
Alexandra also considers two other common utility functions for different investor profiles, though Mean-Variance will be the default for this lab:
    *   **Sortino Reward (downside-only):** Penalizes drawdowns but not upside volatility.
""")
    st.markdown(r"$$ R_t = r_{{p,t}} - \lambda \cdot DD^+_t $$")
    st.markdown(r"where $DD^+_t = \max(0, V_{{\text{{peak}}}} - V_t) / V_{{\text{{peak}}}}$ is the current drawdown relative to the peak.")
    st.markdown(f"""
    *   **Log-Wealth (Kelly-like):** Maximizes geometric growth rate, penalizing large bets.
""")
    st.markdown(r"$$ R_t = \log(1 + r_{{p,t}}) $$")

    st.subheader("Environment Configuration")
    if st.session_state.train_returns is None or st.session_state.train_features is None:
        st.warning("Please load data in '1. Setup & Data' first.")
    else:
        st.session_state.reward_function_type = st.radio(
            "Select Reward Function Type:",
            options=['mean_variance', 'sortino', 'log_wealth'],
            index=0, # Default to mean_variance
            help="Choose the utility function to guide the RL agent's learning."
        )
        st.session_state.risk_aversion_coeff = st.slider(
            "Risk Aversion Coefficient (λ):",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.risk_aversion_coeff,
            step=0.1,
            help="Higher λ means greater risk aversion. Analogous to IPS risk aversion."
        )

        if st.button("Create RL Environment"):
            with st.spinner("Creating portfolio allocation environment..."):
                train_env_instance = PortfolioAllocEnv(
                    returns_df=st.session_state.train_returns,
                    features_df=st.session_state.train_features,
                    initial_value=st.session_state.initial_portfolio_value,
                    tc_rate=st.session_state.transaction_cost_rate,
                    risk_aversion=st.session_state.risk_aversion_coeff,
                    reward_type=st.session_state.reward_function_type
                )
                vec_env_train = DummyVecEnv([lambda: train_env_instance])

                st.session_state.train_env_instance = train_env_instance
                st.session_state.vec_env_train = vec_env_train

            st.success("Portfolio Allocation Environment created successfully!")
            st.write(f"**Reward Type:** `{st.session_state.reward_function_type}` with λ=`{st.session_state.risk_aversion_coeff}`")
            st.write(f"**Observation Space Shape:** {st.session_state.vec_env_train.observation_space.shape}")
            st.write(f"**Action Space Shape:** {st.session_state.vec_env_train.action_space.shape}")
        else:
            if st.session_state.vec_env_train is not None:
                st.info("Environment already created. Re-run to update with new parameters.")
                st.write(f"**Reward Type:** `{st.session_state.reward_function_type}` with λ=`{st.session_state.risk_aversion_coeff}`")
                st.write(f"**Observation Space Shape:** {st.session_state.vec_env_train.observation_space.shape}")
                st.write(f"**Action Space Shape:** {st.session_state.vec_env_train.action_space.shape}")

# --- Page: 3. Train RL Agent ---
elif st.session_state.current_page == "3. Train RL Agent":
    st.header("3. Training the Reinforcement Learning Agent with Proximal Policy Optimization (PPO)")
    st.markdown(f"""
With the environment defined, Alexandra can now train her RL agent. She will use Proximal Policy Optimization (PPO), a robust policy gradient algorithm well-suited for continuous action spaces like asset weight allocation.
""")

    st.subheader("Proximal Policy Optimization (PPO)")
    st.markdown(f"""
Alexandra uses PPO because it's a state-of-the-art, stable, and widely used algorithm for continuous control problems, perfect for learning asset weights directly. Traditional Q-learning algorithms are less suitable as they require discretizing the action space (e.g., specific percentage steps), which can lead to a "curse of dimensionality" for multiple assets.

PPO works by optimizing a "clipped surrogate objective" function, which strikes a balance between making aggressive policy updates and ensuring stability. The objective function is defined as:
""")
    st.markdown(r"$$ L^{{\text{{CLIP}}}}(\theta) = \mathbb{{E}}_t \left[ \min \left( r_t(\theta) \hat{{A}}_t, \text{{clip}}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{{A}}_t \right) \right] $$")
    st.markdown(r"where $r_t(\theta) = \frac{{\pi_\theta(a_t|s_t)}}{{\pi_{{\theta_{{\text{{old}}}}}}(a_t|s_t)}}$ is the probability ratio (how much the new policy $\pi_\theta$ is preferred over the old policy $\pi_{{\theta_{{\text{{old}}}}}}$ for action $a_t$ in state $s_t$), and $\hat{{A}}_t$ is the advantage estimate. The `clip` function limits the policy updates, preventing them from becoming too large and potentially destabilizing the learning process.")
    st.markdown(f"""
The advantage estimate $\hat{{A}}_t$ is typically calculated using Generalized Advantage Estimation (GAE):
""")
    st.markdown(r"$$ \hat{{A}}_t = \sum_{{l=0}}^{{\infty}} (\gamma\lambda)^l \delta_{{t+l}} $$")
    st.markdown(r"where $\delta_t = R_t + \gamma V(s_{{t+1}}) - V(s_t)$ is the Temporal Difference (TD) error.")
    st.markdown(r" - $\gamma$ (discount factor) determines how much future rewards are considered (e.g., 0.99 for long-term investors).")
    st.markdown(r" - $\lambda$ (GAE parameter) balances bias and variance in the advantage estimate (e.g., 0.95 for moderate balance).")
    st.markdown(f"""
These parameters ($\gamma$ and $\lambda$) directly influence the agent's learning horizon and its aggressiveness in incorporating new information, which is relevant for a portfolio manager deciding how quickly to adapt to market shifts.
""")

    st.subheader("Training Configuration")
    if st.session_state.vec_env_train is None:
        st.warning("Please create the RL Environment in '2. Environment Design' first.")
    else:
        st.session_state.total_timesteps = st.number_input(
            "Total Timesteps for Training:",
            min_value=10000,
            max_value=2000000,
            value=st.session_state.total_timesteps,
            step=100000,
            help="Number of interactions the agent has with the environment during training."
        )

        if st.button("Train PPO Agent"):
            with st.spinner(f"Training PPO agent for {st.session_state.total_timesteps} timesteps... This will take a while."):
                ppo_model = train_ppo_agent(
                    st.session_state.vec_env_train,
                    st.session_state.total_timesteps,
                    st.session_state.model_save_path
                )
                st.session_state.ppo_model = ppo_model
            st.success(f"PPO training complete! Model saved to `{st.session_state.model_save_path}`.")
        else:
            if st.session_state.ppo_model is not None:
                st.info("PPO model already trained. Re-train to update with new parameters.")

# --- Page: 4. Baseline Strategies ---
elif st.session_state.current_page == "4. Baseline Strategies":
    st.header("4. Implementing Baseline Portfolio Strategies for Comparison")
    st.markdown(f"""
Before assessing the RL agent's performance, Alexandra needs robust benchmarks. She will implement four traditional portfolio strategies that a typical portfolio manager might use, allowing for an honest comparison of the RL agent's adaptive capabilities.
""")

    st.subheader("Traditional Benchmarks")
    st.markdown(f"""
Alexandra understands that for any new quantitative strategy, a comprehensive comparison against established benchmarks is essential. This helps in understanding the true value-add of the RL approach and identifying scenarios where it outperforms or underperforms. She will implement:

1.  **Equal-Weight (EW):** A simple strategy where all assets are equally weighted and rebalanced daily. This represents a naive diversification.
2.  **Fixed 60/40 Stock-Bond:** A classic strategic asset allocation, rebalanced monthly. This is a common institutional benchmark. For this portfolio, Alexandra defines the fixed weights as 60% SPY, 30% TLT, 5% GLD, and 5% SHY.
3.  **Static Markowitz Optimization:** Weights are optimized once at the beginning of the test period using historical data (e.g., the training period) to maximize the Sharpe ratio (or minimize variance for a target return) and remain fixed thereafter. This is the cornerstone of modern portfolio theory, but it's static. For simplicity and to show a static allocation, we'll compute minimum variance weights based on the training data.
4.  **Monthly Rebalanced 60/40:** A tactical strategy that rebalances back to a 60/40 target (60% SPY, 30% TLT, 5% GLD, 5% SHY) at the start of each month, allowing for drift within the month. This introduces periodic rebalancing without dynamic adaptation to market regimes.
""")

    if st.session_state.test_returns is None or st.session_state.train_returns is None:
        st.warning("Please load data in '1. Setup & Data' first.")
    else:
        # Re-compute static Markowitz weights using current train_returns
        # This part of the source.py needs to be called to get the static_markowitz_fn
        # However, the source.py directly computes `mv_weights_static` global to the script.
        # I need to ensure this is handled for the `run_baseline_strategy` function.
        # Given the instruction "import and use those functions directly — do not redefine, rewrite, stub, or duplicate them."
        # I'll assume the global variables/functions are accessible.

        if st.button("Run Baseline Strategies"):
            with st.spinner("Simulating baseline strategies..."):
                baseline_results = {}
                # The BASELINE_STRATEGIES dict is defined in source.py.
                # Accessing global constants/functions from source.py.
                # Ensure the parameters are passed from session_state.
                for name, fn in BASELINE_STRATEGIES.items():
                    if name == 'Fixed 60/40': # Use monthly_rebal_60_40_fn for fixed 60/40 target with monthly rebalance
                        values, weights_hist = run_baseline_strategy(
                            st.session_state.test_returns,
                            monthly_rebal_60_40_fn, # This function needs to be explicitly available from source
                            name,
                            st.session_state.initial_portfolio_value,
                            st.session_state.transaction_cost_rate
                        )
                    else:
                        values, weights_hist = run_baseline_strategy(
                            st.session_state.test_returns,
                            fn,
                            name,
                            st.session_state.initial_portfolio_value,
                            st.session_state.transaction_cost_rate
                        )
                    baseline_results[name] = {'values': values, 'weights': weights_hist}
                st.session_state.baseline_results = baseline_results
            st.success("All baseline strategies simulated successfully!")
        else:
            if st.session_state.baseline_results is not None:
                st.info("Baseline strategies already simulated.")

# --- Page: 5. Evaluate Performance ---
elif st.session_state.current_page == "5. Evaluate Performance":
    st.header("5. Evaluating the RL Agent and Baselines on Out-of-Sample Data")
    st.markdown(f"""
Now, Alexandra will evaluate the trained RL agent on the unseen test dataset and compare its performance metrics (Sharpe Ratio, Max Drawdown, Total Return, Turnover) against the established baseline strategies. This is the ultimate test of the RL agent's adaptiveness and robustness in real-world market conditions.
""")

    st.subheader("Performance Metrics")
    st.markdown(f"""
For Alexandra, performance evaluation is not just about raw returns. Risk-adjusted returns (Sharpe Ratio), downside risk (Max Drawdown), and trading efficiency (Turnover) are equally, if not more, important. High turnover can erode returns through transaction costs, making a strategy impractical. The test period (2020-2024) is deliberately chosen to include significant market shocks (COVID-19, 2022 rate hikes), providing a rigorous test of the agent's ability to navigate volatile regimes.
""")
    st.markdown(r"$$ \text{{Sharpe Ratio}} = \frac{{\text{{Mean(Portfolio Returns) - Risk-Free Rate}}}}{{\text{{Std Dev(Portfolio Returns)}}}} \times \sqrt{{\text{{Annualization Factor}}}} $$")
    st.markdown(r"Here, we simplify by assuming a 0% risk-free rate, and the annualization factor is $\sqrt{{252}}$ for daily data.")
    st.markdown(r"$$ \text{{Max Drawdown}} = \max_{{t}} \left( 1 - \frac{{V_t}}{{\max_{{s \le t}} V_s}} \right) $$")
    st.markdown(r"where $V_t$ is the portfolio value at time $t$, and $\max_{{s \le t}} V_s$ is the peak portfolio value up to time $t$.")
    st.markdown(r"$$ \text{{Turnover}} = \frac{{1}}{{T}} \sum_{{t=1}}^{{T}} ||\mathbf{{w}}_t - \mathbf{{w}}_{{t-1}}||_1 $$")
    st.markdown(r"where $\mathbf{{w}}_t$ is the weight vector at time $t$, and $T$ is the number of rebalancing periods. For daily rebalancing, this is the average daily L1 norm of weight changes.")

    if st.session_state.ppo_model is None or st.session_state.baseline_results is None or st.session_state.test_returns is None or st.session_state.test_features is None:
        st.warning("Please ensure data is loaded, environment is created, RL agent is trained, and baseline strategies are run.")
    else:
        if st.button("Evaluate All Strategies"):
            with st.spinner("Evaluating RL Agent and Baseline Strategies..."):
                # --- Evaluate RL Agent ---
                test_env_instance = PortfolioAllocEnv(
                    returns_df=st.session_state.test_returns,
                    features_df=st.session_state.test_features,
                    initial_value=st.session_state.initial_portfolio_value,
                    tc_rate=st.session_state.transaction_cost_rate,
                    risk_aversion=st.session_state.risk_aversion_coeff,
                    reward_type=st.session_state.reward_function_type
                )
                obs, info = test_env_instance.reset()

                rl_values = [info['portfolio_value']]
                rl_weights_hist = [info['weights']]

                done = False
                while not done:
                    action, _ = st.session_state.ppo_model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = test_env_instance.step(action)
                    rl_values.append(info['portfolio_value'])
                    rl_weights_hist.append(info['weights'])

                # Remove the initial equal weight from rl_weights_hist as it's not an action of the agent
                rl_weights_hist = rl_weights_hist[1:]

                rl_metrics = evaluate_strategy_performance(
                    rl_values,
                    st.session_state.initial_portfolio_value,
                    rl_weights_hist,
                    st.session_state.transaction_cost_rate
                )

                all_strategy_metrics = {'RL Agent (PPO)': rl_metrics}
                strategy_equity_curves = {'RL Agent (PPO)': rl_values}

                # --- Evaluate Baseline Strategies ---
                for name, results in st.session_state.baseline_results.items():
                    metrics = evaluate_strategy_performance(
                        results['values'],
                        st.session_state.initial_portfolio_value,
                        results['weights'],
                        st.session_state.transaction_cost_rate
                    )
                    all_strategy_metrics[name] = metrics
                    strategy_equity_curves[name] = results['values']

                st.session_state.rl_values = rl_values
                st.session_state.rl_weights_hist = rl_weights_hist
                st.session_state.all_strategy_metrics = all_strategy_metrics
                st.session_state.strategy_equity_curves = strategy_equity_curves

            st.success("Performance evaluation complete!")
            metrics_df = pd.DataFrame(st.session_state.all_strategy_metrics).T
            st.dataframe(metrics_df.style.format({
                'Total Return': "{:.2%}",
                'Sharpe Ratio': "{:.3f}",
                'Max Drawdown': "{:.1%}",
                'Annualized Turnover': "{:.2f}"
            }))
        else:
            if st.session_state.all_strategy_metrics is not None:
                st.info("Performance metrics already calculated.")
                metrics_df = pd.DataFrame(st.session_state.all_strategy_metrics).T
                st.dataframe(metrics_df.style.format({
                    'Total Return': "{:.2%}",
                    'Sharpe Ratio': "{:.3f}",
                    'Max Drawdown': "{:.1%}",
                    'Annualized Turnover': "{:.2f}"
                }))

# --- Page: 6. Policy Insights ---
elif st.session_state.current_page == "6. Policy Insights":
    st.header("6. Visualizing Policy Behavior and Performance Insights")
    st.markdown(f"""
To gain deeper insights, Alexandra will visualize the performance and the RL agent's learned allocation policy. This helps in understanding *when* and *how* the agent adapts to different market conditions, which is crucial for building trust and explainability for stakeholders.
""")

    st.subheader("Visualizations for Policy Interpretation")
    st.markdown(f"""
Visualizations are essential for Alexandra to communicate the strategy's efficacy and interpret its behavior.
*   **Equity Curves:** Show the cumulative wealth growth of each strategy, providing a high-level performance comparison.
*   **Stacked Weight Allocation (RL Agent):** This heatmap-like visualization (stacked area chart) reveals the dynamic shifts in the RL agent's portfolio weights over time. Alexandra can correlate these shifts with market events (e.g., increased gold allocation during volatility spikes, decreased equities during downturns), demonstrating the "regime-adaptive" nature of the policy. This is the core "policy interpretation" tool.
*   **Market Volatility Context:** Plotting rolling market volatility alongside the allocation shifts provides context, allowing Alexandra to see if the agent's rebalancing aligns with risk-on/risk-off indicators.
*   **Performance Comparison Bar Chart:** A direct visual comparison of key metrics (Sharpe, Max Drawdown, Total Return, Turnover) helps quickly identify strengths and weaknesses.
""")
    st.markdown(f"---")
    st.header("Analyzing Regime-Conditional Allocation and Transaction Cost Efficiency")
    st.markdown(f"""
Alexandra concludes her analysis by examining the RL agent's behavior during specific market regimes and assessing the impact of transaction costs. This critical step provides deeper insights into the agent's intelligence and its practical applicability.
""")
    st.subheader("Regime-Conditional Analysis")
    st.markdown(f"""
A key advantage of dynamic allocation is its ability to adapt to changing market conditions. Alexandra wants to explicitly see if the RL agent exhibited "risk-on" or "risk-off" behavior, for instance, by reducing equity exposure and increasing safe-haven assets during high-volatility periods. This helps validate the agent's learning process.

Furthermore, transaction costs are a *reality check* for any high-frequency rebalancing strategy. If an agent performs well but incurs massive turnover, its net performance will be severely eroded in practice. Alexandra needs to understand the trade-off the RL agent makes between performance and trading costs.
""")

    if st.session_state.all_strategy_metrics is None or st.session_state.strategy_equity_curves is None or st.session_state.rl_weights_hist is None or st.session_state.test_returns is None or st.session_state.test_features is None:
        st.warning("Please complete previous steps: Data Loading, Environment Creation, RL Training, Baseline Simulations, and Performance Evaluation.")
    else:
        if st.button("Generate Performance Plots & Analysis"):
            with st.spinner("Generating visualizations and performing analysis..."):
                metrics_df = pd.DataFrame(st.session_state.all_strategy_metrics).T
                
                # --- Visualization 1: Three-Panel Dashboard ---
                fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})

                # Panel 1: Equity Curves
                for name, values in st.session_state.strategy_equity_curves.items():
                    ax1.plot(st.session_state.test_returns.index[:len(values)], values, label=f'{name} (Sharpe={st.session_state.all_strategy_metrics[name]["Sharpe Ratio"]:.2f})')
                ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
                ax1.set_title('Dynamic Asset Allocation: RL Agent vs. Baselines - Equity Curves', fontsize=14)
                ax1.legend(loc='upper left', fontsize=10)
                ax1.grid(True, linestyle='--', alpha=0.6)
                ax1.tick_params(axis='x', rotation=45)

                # Panel 2: Stacked Area Chart of RL Agent's Asset Weight Allocations
                # Need ASSET_TICKERS from source.py or session_state.tickers_dict
                rl_weights_df = pd.DataFrame(st.session_state.rl_weights_hist, columns=list(st.session_state.tickers_dict.values()), index=st.session_state.test_returns.index[:len(st.session_state.rl_weights_hist)])
                colors = ['#2196F3', '#FF9800', '#FFD700', '#9E9E9E'] # Blue (SPY), Orange (TLT), Gold (GLD), Grey (SHY)
                ax2.stackplot(rl_weights_df.index, rl_weights_df.T, labels=rl_weights_df.columns, colors=colors, alpha=0.8)
                ax2.set_ylabel('Portfolio Weight', fontsize=12)
                ax2.set_ylim(0, 1)
                ax2.set_title('RL Agent: Learned Allocation Over Time', fontsize=14)
                ax2.legend(loc='upper right', fontsize=10)
                ax2.grid(True, linestyle='--', alpha=0.6)
                ax2.tick_params(axis='x', rotation=45)
                st.session_state.rl_weights_df = rl_weights_df # Store for regime analysis

                # Panel 3: Rolling Volatility for Market Context
                rolling_vol = st.session_state.test_returns['Equities'].rolling(21).std() * np.sqrt(252)
                ax3.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.3, color='red')
                ax3.set_ylabel('Equity Volatility (Ann.)', fontsize=12)
                ax3.set_title('Market Context: Equity Realized Volatility', fontsize=14)
                ax3.grid(True, linestyle='--', alpha=0.6)
                ax3.tick_params(axis='x', rotation=45)

                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

                # --- Visualization 2: Bar Chart Comparing Performance Metrics ---
                fig2, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=False)
                metrics_to_plot = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Annualized Turnover']
                plot_colors = ['skyblue', 'lightcoral', 'lightgreen', 'mediumpurple', 'gold', 'teal']

                for i, metric in enumerate(metrics_to_plot):
                    ax = axes[i]
                    metric_values = metrics_df[metric]
                    metric_values.plot(kind='bar', ax=ax, color=plot_colors[:len(metric_values)], legend=False)
                    ax.set_title(metric, fontsize=14)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)

                    if 'Return' in metric or 'Drawdown' in metric:
                        vals = ax.get_yticks()
                        ax.set_yticklabels([f'{x:,.1%}' for x in vals])
                    elif 'Turnover' in metric:
                        vals = ax.get_yticks()
                        ax.set_yticklabels([f'{x:,.0f}%' for x in vals])
                    else:
                        ax.tick_params(axis='y')

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

                # --- Regime-Conditional Allocation Analysis ---
                st.subheader("Regime-Conditional Allocation")
                regime_allocations = analyze_regime_conditional_allocation(st.session_state.rl_weights_df, st.session_state.test_features)
                st.session_state.regime_allocations = regime_allocations
                st.write("Average RL Agent Weights by Market Volatility Regime:")
                st.dataframe(regime_allocations.style.format(lambda x: f"{x:.2%}"))

                # --- Transaction Cost & Turnover Efficiency Analysis ---
                st.subheader("Transaction Cost & Turnover Efficiency")
                turnover_comparison = compare_turnover_efficiency(st.session_state.all_strategy_metrics)
                st.session_state.turnover_comparison = turnover_comparison
                st.write("Annualized Turnover Comparison (Higher is more trading):")
                st.dataframe(turnover_comparison.style.format(lambda x: f"{x:.2f}"))

            st.success("All plots and analyses generated!")
        else:
            if st.session_state.regime_allocations is not None:
                st.info("Plots and analyses already generated.")
                st.subheader("Regime-Conditional Allocation")
                st.write("Average RL Agent Weights by Market Volatility Regime:")
                st.dataframe(st.session_state.regime_allocations.style.format(lambda x: f"{x:.2%}"))
                st.subheader("Transaction Cost & Turnover Efficiency")
                st.write("Annualized Turnover Comparison (Higher is more trading):")
                st.dataframe(st.session_state.turnover_comparison.style.format(lambda x: f"{x:.2f}"))
```
