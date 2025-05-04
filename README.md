# Adaptive-Sector-Rotation-Framework


An end-to-end Python toolkit combining advanced feature engineering, machine-learning regressors, explainability (SHAP, LIME) and deep Q-network (DQN) reinforcement learning for dynamic sector allocation.

## 📚 Project Motivation

Traditional sector rotation relies on simple linear models that struggle with financial markets’ complex, non-linear interactions. This framework elevates rotation strategies by:
- Engineering hundreds of **lagged**, **rolling** and **interaction** macro features  
- Training six ML regressors (RF, XGBoost, LightGBM, SVM, Elastic Net, MLP) to predict sector returns  
- Applying **SHAP** and **LIME** for global & local interpretability  
- Stacking base models via **Optuna-tuned** ensemble weights  
- Framing allocation as a sequential decision problem solved by a **Deep Q-Network** (DQN) with experience replay and ε-greedy exploration  
- Backtesting dynamic allocations against benchmarks using a vectorized simulator  

This hybrid ML+RL pipeline outperforms static benchmarks (S&P 500) over 2000–2024, delivering higher Sharpe ratios and controlled drawdowns
