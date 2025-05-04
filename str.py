import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from collections import deque

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split, TimeSeriesSplit, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import lime.lime_tabular
import shap
import plotly.express as px
import plotly.graph_objects as go

# Global settings and random seeds for reproducibility
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')
sns.set(style="whitegrid")
pd.set_option('display.precision', 8)
np.set_printoptions(precision=8)


###############################################################################
#                            DATA PROCESSING CLASS                            #
###############################################################################
class DataProcessor:
    """
    Processes and engineers features from sector and macroeconomic data.
    """
    def __init__(self):
        pass

    def compute_returns(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns for sector data.
        """
        sector_data['date'] = pd.to_datetime(sector_data['date'])
        sector_columns = sector_data.columns.difference(['date'])
        sector_returns = np.log(sector_data[sector_columns] / sector_data[sector_columns].shift(1))
        sector_returns.bfill(inplace=True, axis=0)
        sector_returns['date'] = sector_data['date']
        return sector_returns

    def preprocess_macro(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess macroeconomic data by converting date column to datetime.
        """
        macro_data['date'] = pd.to_datetime(macro_data['date'])
        return macro_data

    def future_engineering(self, data: pd.DataFrame, lags: list, rolling_windows: list = [3, 6, 9]) -> pd.DataFrame:
        """
        Create rolling statistics, lagged features, and interaction terms.
        """
        columns = data.columns.difference(['date'])
        for col in data.columns:
            if col != 'date':
                for window in rolling_windows:
                    data[f"{col}_roll_mean{window}"] = data[col].rolling(window).mean()
                    data[f"{col}_roll_std{window}"] = data[col].rolling(window).std()

        lagged_data = {}
        for col in columns:
            for lag in lags:
                lagged_data[f"{col}_lag{lag}"] = data[col].shift(lag)
        lagged_df = pd.concat([data, pd.DataFrame(lagged_data)], axis=1)

        numeric_cols = lagged_df.select_dtypes(include=[np.number]).columns
        lagged_df[numeric_cols] = lagged_df[numeric_cols].interpolate(method='linear', limit_direction='both')

        # Compute correlation matrix and interaction terms from top pairs
        corr_matrix = data[columns].corr()
        corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
        top_corr_pairs = corr_pairs.dropna().head(2).index

        for col1, col2 in top_corr_pairs:
            lagged_df[f"{col1}_x_{col2}"] = data[col1] * data[col2]

        interaction_pairs = [
            ('CPI', 'Consumer Confidence Index'),
            ('N-F Payrolls', 'Jobless Claims'),
            ('Personal Saving Rate', 'Retail Sales'),
        ]
        for col1, col2 in interaction_pairs:
            if col1 in lagged_df.columns and col2 in lagged_df.columns:
                lagged_df[f"{col1}_x_{col2}"] = lagged_df[col1] * lagged_df[col2]

        return lagged_df

    def rate_of_change(self, data: pd.DataFrame, period: int = 1) -> pd.DataFrame:
        """
        Compute the rate of change (ROC) for each column.
        """
        columns = data.columns.difference(['date'])
        roc_dict = {f"{col}_ROC": (data[col] / data[col].shift(period)) - 1 for col in columns}
        roc_data = pd.concat(roc_dict, axis=1)
        numeric_cols = roc_data.select_dtypes(include=[np.number]).columns
        roc_data[numeric_cols] = roc_data[numeric_cols].interpolate(method='linear', limit_direction='both', order=2)
        roc_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        roc_data.interpolate(method='linear', limit_direction='both', inplace=True)
        return roc_data


###############################################################################
#                             MODEL UTILITIES                                 #
###############################################################################
class ModelUtils:
    """
    Utility methods for model parameter adjustments and metric calculations.
    """
    # Dictionary to map model names to classes
    model_classes = {
        'random_forest': RandomForestRegressor,
        'xgboost': XGBRegressor,
        'lgbm': LGBMRegressor,
    }

    @staticmethod
    def adjust_params(model_name: str, params: dict) -> dict:
        """
        Adjust and round model parameters that must be integers.
        """
        adjusted_params = params.copy()
        if model_name in ['random_forest', 'xgboost', 'lgbm']:
            int_params = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'num_leaves']
        elif model_name == 'mlp':
            int_params = ['max_iter']
        elif model_name == 'elastic_net':
            int_params = ['max_iter']
        else:
            int_params = []

        for param in int_params:
            if param in adjusted_params and not isinstance(adjusted_params[param], tuple):
                adjusted_params[param] = int(round(adjusted_params[param]))
        return adjusted_params

    @staticmethod
    def calculate_regression_metrics(y_true: np.array, y_pred: np.array) -> dict:
        """
        Calculate common regression metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        correct_directions = np.sum((np.sign(y_true) == np.sign(y_pred)))
        total = len(y_true)
        directional_accuracy = (correct_directions / total) * 100
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Directional Accuracy": directional_accuracy
        }


###############################################################################
#                                 PLOTTING                                    #
###############################################################################
class Plotter:
    """
    Collection of static methods to produce various plots.
    """
    @staticmethod
    def plot_feature_importance(model, feature_names: list, model_name: str, sector: str):
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            sorted_idx = np.argsort(importance)[-10:]
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_idx)), importance[sorted_idx], edgecolor="k")
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel("Feature Importance")
            plt.title(f"Top 10 Feature Importances - {model_name} for {sector}")
            plt.gca().invert_yaxis()
            plt.show()

    @staticmethod
    def plot_shap_values(shap_values, X_test, model_name: str, sector: str):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=10, show=False)
        plt.title(f"SHAP Summary Plot - {model_name} for {sector}")
        plt.show()

    @staticmethod
    def plot_lime_explanation(model, X_test, model_name: str, sector: str):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_test),
            feature_names=X_test.columns,
            mode='regression'
        )
        instance_index = X_test.shape[0] // 2
        explanation = explainer.explain_instance(X_test.iloc[instance_index], model.predict, num_features=10)
        explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation - {model_name} for {sector}")
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def plot_learning_curve(estimator, X, y, title="Learning Curve"):
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=TimeSeriesSplit(n_splits=5),
                                                                scoring='neg_mean_squared_error')
        train_mean = -train_scores.mean(axis=1)
        test_mean = -test_scores.mean(axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, marker='o', label="Training Error")
        plt.plot(train_sizes, test_mean, marker='o', label="Validation Error")
        plt.xlabel("Training Set Size")
        plt.ylabel("MSE")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(f"learning_curve_{title}.png")
        plt.close()

    @staticmethod
    def plot_error_distribution(residuals, model_name: str, sector: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title(f"Error Distribution for {model_name} - {sector}")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()
        plt.savefig(f"error_distribution_{model_name}_{sector}.png")
        plt.close()

    @staticmethod
    def plot_weighted_contribution(best_weights: dict):
        plt.figure(figsize=(8, 6))
        model_names = list(best_weights.keys())
        weights = list(best_weights.values())
        plt.barh(model_names, weights, edgecolor="black")
        plt.xlabel("Optimized Weight")
        plt.ylabel("Model")
        plt.title("Weighted Contribution of Each Base Model")
        plt.show()

    @staticmethod
    @staticmethod
    def plot_cumulative_returns(backtest_results: list, dates: pd.Series):
        plt.figure(figsize=(18, 9))
        colors = sns.color_palette("tab10", len(backtest_results))
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        for idx, result in enumerate(backtest_results):
            returns = pd.Series(result['returns over time'])
            returns = np.exp(returns) - 1  # Transform log returns to normal returns
            cumulative_returns = (1 + returns).cumprod() - 1
            plot_dates = dates[:len(cumulative_returns)]
            annualized_return = np.mean(returns)
            annualized_volatility = np.std(returns)
            sharpe_ratio = annualized_return / (annualized_volatility + 1e-10)
            plt.plot(plot_dates, cumulative_returns,
                     label=f"{result['model']} - Sharpe: {sharpe_ratio:.2f}",
                     color=colors[idx],
                     linestyle=line_styles[idx % len(line_styles)],
                     linewidth=2)
        plt.title("Cumulative Returns for All Models")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_drawdown_fixed(backtest_results: list, dates: pd.Series):
        plt.figure(figsize=(18, 9))
        colors = sns.color_palette("tab20", len(backtest_results))
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        for idx, result in enumerate(backtest_results):
            returns = pd.Series(result['returns over time'])
            cumulative_returns = (1 + returns).cumprod()
            cumulative_max = cumulative_returns.cummax()
            drawdown = (cumulative_max - cumulative_returns) / cumulative_max
            drawdown = -drawdown
            plot_dates = dates[:len(drawdown)]
            plt.plot(plot_dates, drawdown,
                     label=f"{result['model']}",
                     color=colors[idx],
                     linestyle=line_styles[idx % len(line_styles)],
                     linewidth=2)
        plt.title("Drawdown Plot for Each Model")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_volatility_vs_return(backtest_results: list):
        metrics = []
        for result in backtest_results:
            # Calculate metrics (simplified example using dummy computations)
            # In practice, calculate annualized return and volatility from returns.
            annualized_return = np.mean(result['returns over time'])
            annualized_volatility = np.std(result['returns over time'])
            sharpe_ratio = annualized_return / (annualized_volatility + 1e-10)
            max_drawdown = np.min(result['returns over time'])
            metrics.append((result['model'], annualized_return, annualized_volatility, sharpe_ratio, max_drawdown))
        metrics_df = pd.DataFrame(metrics, columns=['Model', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown'])
        fig = px.scatter(metrics_df, x='Annualized Volatility', y='Annualized Return',
                         size='Sharpe Ratio', color='Model', hover_data=['Max Drawdown'],
                         text='Model', title="Volatility vs. Return")
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_title="Annualized Volatility", yaxis_title="Annualized Return")
        fig.show()


###############################################################################
#                              MODEL TRAINER                                  #
###############################################################################
class ModelTrainer:
    """
    Trains and evaluates base models for each sector.
    """
    def __init__(self, sector_returns: pd.DataFrame, macro_data: pd.DataFrame, hyperparameters: pd.DataFrame, rfe_features: dict):
        self.sector_returns = sector_returns
        self.macro_data = macro_data
        self.hyperparameters = hyperparameters
        self.rfe_features = rfe_features

    def train_and_evaluate_models(self) -> tuple:
        """
        Train models for each sector and collect performance metrics, feature importance, and SHAP values.
        Returns a tuple containing:
          - results: Dictionary with residuals and SHAP values
          - model_performance: Dictionary with metrics per sector/model
          - feature_importance_data: List of feature importance records
          - trained_models: Dictionary with models per sector
          - features: Dictionary with selected features per sector
        """
        results = {}
        model_performance = {}
        feature_importance_data = []
        trained_models = {}
        features = {}

        for sector in self.sector_returns.columns.difference(['date']):
            X = self.macro_data.drop(columns=['date'])
            y = self.sector_returns[sector]
            print(f"Training models for sector: {sector}")

            if sector in self.rfe_features:
                selected_features = self.rfe_features[sector]
            else:
                raise ValueError(f"No RFE-selected features for sector: {sector}")

            X_train, X_test, y_train, y_test = train_test_split(
                X[selected_features], y, test_size=0.25, shuffle=False, random_state=RANDOM_SEED)

            sector_results = {'residuals': {}, 'shap_values': {}}
            model_metrics = {}
            sector_models = []

            hyper_sector = self.hyperparameters[self.hyperparameters['sector'] == sector]
            for _, row in hyper_sector.iterrows():
                model_name = row['model']
                # Evaluate string parameters safely (assumes trusted input)
                model_params = eval(row['parameters'])
                model_params = ModelUtils.adjust_params(model_name, model_params)
                model_class = ModelUtils.model_classes.get(model_name)
                if model_class:
                    model = model_class(**model_params)
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error")
                    print(f"{model_name} CV MSE for {sector}: {np.mean(-cv_scores):.4f}")
                    model.fit(X_train, y_train)
                    sector_models.append(model)
                    predictions = model.predict(X_test)
                    residuals = y_test - predictions
                    sector_results['residuals'][model_name] = residuals
                    metrics = ModelUtils.calculate_regression_metrics(y_test, predictions)
                    model_metrics[model_name] = metrics
                    print(f"{model_name} Metrics for {sector}: {metrics}")

                    if hasattr(model, "feature_importances_"):
                        importance = model.feature_importances_
                        for i, feature in enumerate(selected_features):
                            feature_importance_data.append({
                                "Sector": sector,
                                "Model": model_name,
                                "Feature": feature,
                                "Importance": importance[i]
                            })
                        # Compute SHAP values using an appropriate explainer
                        explainer = None
                        if model_name in ['random_forest', 'xgboost', 'lgbm']:
                            explainer = shap.TreeExplainer(model)
                        elif model_name == 'elastic_net':
                            explainer = shap.LinearExplainer(model, X_train)
                        elif model_name in ['svm', 'mlp']:
                            explainer = shap.KernelExplainer(model.predict, X_train)
                        if explainer:
                            subset_size = min(100, len(X_test))
                            shap_values = explainer.shap_values(X_test[:subset_size])
                            sector_results['shap_values'][model_name] = shap_values
            features[sector] = selected_features
            results[sector] = sector_results
            model_performance[sector] = model_metrics
            trained_models[sector] = sector_models

        return results, model_performance, feature_importance_data, trained_models, features


###############################################################################
#                             ENSEMBLE TRAINER                                #
###############################################################################
class EnsembleTrainer:
    """
    Implements stacking ensemble methods and weight optimization.
    """
    def get_base_model_predictions(self, trained_models: dict, X: pd.DataFrame, y: pd.Series, cv: int = 2) -> pd.DataFrame:
        kf = KFold(n_splits=cv, shuffle=False)
        oof_predictions = pd.DataFrame(index=X.index)
        for model_name, model in trained_models.items():
            print(f"Generating OOF predictions for {model_name}...")
            oof_pred = cross_val_predict(model, X, y, cv=kf)
            oof_predictions[model_name] = oof_pred
        return oof_predictions

    def optimize_weights(self, X_stack_train: pd.DataFrame, y_train: pd.Series) -> dict:
        model_columns = X_stack_train.columns

        def objective(trial):
            weights = {col: trial.suggest_float(f"{col}", 0.001, 1.0) for col in model_columns}
            weighted_pred = np.sum([weights[col] * X_stack_train[col] for col in model_columns], axis=0)
            mse = mean_squared_error(y_train, weighted_pred)
            return mse

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=20)
        best_weights = study.best_params
        total_weight = sum(best_weights.values())
        if total_weight == 0:
            print("Warning: All weights zero; using equal weights.")
            best_weights = {col: 1.0 / len(model_columns) for col in model_columns}
        else:
            best_weights = {col: weight / total_weight for col, weight in best_weights.items()}
        return best_weights

    def train_meta_model(self, X_stack_train: pd.DataFrame, y_train: pd.Series, meta_model_type="RandomForest"):
        def objective(trial):
            if meta_model_type == "Ridge":
                alpha = trial.suggest_float("alpha", 0.1, 10.0, log=True)
                meta_model = Ridge(alpha=alpha)
            elif meta_model_type == "GradientBoosting":
                n_estimators = trial.suggest_int("n_estimators", 50, 200)
                max_depth = trial.suggest_int("max_depth", 2, 6)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                meta_model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
            elif meta_model_type == "LGBM":
                num_leaves = trial.suggest_int("num_leaves", 10, 150)
                learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5, log=True)
                n_estimators = trial.suggest_int("n_estimators", 50, 600)
                meta_model = LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators, verbose=-1)
            elif meta_model_type == "RandomForest":
                n_estimators = trial.suggest_int("n_estimators", 50, 600)
                max_depth = trial.suggest_int("max_depth", 2, 15)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
                meta_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            else:
                raise ValueError("Unsupported meta_model_type")
            meta_model.fit(X_stack_train, y_train)
            preds = meta_model.predict(X_stack_train)
            mse = mean_squared_error(y_train, preds)
            return mse

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=40)
        best_params = study.best_params

        if meta_model_type == "Ridge":
            meta_model = Ridge(**best_params)
        elif meta_model_type == "GradientBoosting":
            meta_model = GradientBoostingRegressor(**best_params)
        elif meta_model_type == "LGBM":
            meta_model = LGBMRegressor(**best_params)
        elif meta_model_type == "RandomForest":
            meta_model = RandomForestRegressor(**best_params)
        else:
            raise ValueError("Unsupported meta_model_type")
        meta_model.fit(X_stack_train, y_train)
        return meta_model

    def train_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  sector_models: dict, meta_model_type="RandomForest"):
        # Ensure sector_models is a dictionary; if provided as list, assign default names
        if isinstance(sector_models, list):
            model_names = ['random_forest', 'xgboost', 'lgbm', 'svm', 'elastic_net', 'mlp']
            sector_models = {model_names[i]: model for i, model in enumerate(sector_models)}
        else:
            model_names = list(sector_models.keys())
        X_stack_train = self.get_base_model_predictions(sector_models, X_train, y_train)
        X_stack_test = self.get_base_model_predictions(sector_models, X_test, y_test)
        best_weights = self.optimize_weights(X_stack_train, y_train)
        weighted_train_preds = np.sum(
            [best_weights.get(model_name, 0) * X_stack_train[model_name] for model_name in X_stack_train.columns],
            axis=0)
        weighted_test_preds = np.sum(
            [best_weights.get(model_name, 0) * X_stack_test[model_name] for model_name in X_stack_test.columns],
            axis=0)
        meta_model = self.train_meta_model(X_stack_train, y_train, meta_model_type)
        meta_predictions = meta_model.predict(X_stack_test)
        meta_model_metrics = ModelUtils.calculate_regression_metrics(y_test, meta_predictions)
        print(f"Stacked Model Metrics: {meta_model_metrics}")
        return meta_model, meta_predictions, X_stack_test, best_weights, meta_model_metrics


###############################################################################
#                            DQN ALLOCATOR (RL)                               #
###############################################################################

class DQNAllocator:
    """
    Deep Q-Network (DQN) for sector allocation with experience replay.
    """
    def __init__(self, state_size: int, action_size: int, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, learning_rate: float = 0.0001,
                 memory_maxlen: int = 1500, seed: int = RANDOM_SEED):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_maxlen)
        self.prioritized_memory = []
        np.random.seed(seed)
        random.seed(seed)
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool):
        # Ensure states are 1D arrays of the expected size.
        expected = self.state_size
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        if state.shape[0] != expected:
            state = np.zeros(expected)
        if next_state.shape[0] != expected:
            next_state = np.zeros(expected)
        td_error = abs(
            reward + self.gamma * np.max(self.model.predict(next_state.reshape(1, -1))) -
            np.max(self.model.predict(state.reshape(1, -1)))
        )
        self.prioritized_memory.append((state, action, reward, next_state, done, td_error))
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        # Use random.sample to avoid forcing a homogeneous numpy array.
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([np.asarray(x[0]).flatten() for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([np.asarray(x[3]).flatten() for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        target_qs = self.model.predict_on_batch(states)
        next_qs = self.model.predict_on_batch(next_states)

        for i in range(batch_size):
            if dones[i]:
                target_qs[i, actions[i]] = rewards[i]
            else:
                target_qs[i, actions[i]] = rewards[i] + self.gamma * np.max(next_qs[i])
        self.model.fit(states, target_qs, epochs=1, verbose=0, batch_size=batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state: np.array) -> int:
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"Random action chosen: {action}")
            return action
        act_values = self.model.predict(state.reshape(1, -1))
        action = np.argmax(act_values[0])
        print(f"Predicted action: {action} from {act_values}")
        return action




###############################################################################
#                               BACKTESTER                                    #
###############################################################################
class Backtester:
    """
    Backtesting routines for dynamic allocation using RL and meta-models.
    """

    def compute_reward(self, portfolio_return: float, portfolio_returns_history: list) -> float:
        """
        Compute reward based on the Sharpe ratio.
        """
        risk_free_rate = 0
        if len(portfolio_returns_history) == 0:
            return 0
        mean_return = np.mean(portfolio_returns_history)
        excess_return = mean_return - risk_free_rate
        volatility = np.std(portfolio_returns_history)
        sharpe_ratio = excess_return / (volatility + 1e-12)
        return sharpe_ratio

    def vectorized_backtest(self, sector_returns: pd.DataFrame, merged_macro: pd.DataFrame,
                            trained_models: dict, futures: dict, model_names: list,
                            action_size: int, weights: dict, shap_values_cache: dict,
                            window_size: int, step: int, batch_size: int = 45) -> tuple:
        """
        Run a vectorized backtest for dynamic allocation.
        This version:
          - Persists a DQNAllocator per (sector, model) pair.
          - Uses weighted state inputs derived from SHAP values.
          - Ensures all states are 1D arrays of the expected length.
        """
        # List of sectors (excluding 'date')
        sectors = list(sector_returns.columns.difference(['date']))

        # Create persistent DQNAllocator agents per sector and model:
        persistent_agents = {sector: {} for sector in sectors}
        for sector in sectors:
            for model_name in model_names:
                state_size_agent = len(shap_values_cache[sector][model_name]["features"])
                persistent_agents[sector][model_name] = DQNAllocator(state_size=state_size_agent,
                                                                     action_size=action_size)

        dynamic_weights_over_time = {model_name: [] for model_name in model_names}
        meta_model_allocations = []
        rewards_over_episodes = []
        sharpe_ratios_over_episodes = []

        # Loop over time windows
        for start in range(0, len(merged_macro) - window_size, step):
            end = start + window_size
            current_macro_window = merged_macro.iloc[start:end].copy()
            current_sector_window = sector_returns.iloc[start:end].copy()
            cumulative_rewards = 0
            episode_returns = []

            for model_name in model_names:
                model_sector_weights = []
                for t in range(len(current_macro_window)):
                    sector_allocations = []
                    for sector in sectors:
                        agent = persistent_agents[sector][model_name]
                        cache = shap_values_cache[sector][model_name]
                        relevant_features = cache["features"]
                        feature_weights = cache["weights"]
                        raw_state = current_macro_window[relevant_features].iloc[t].values
                        # Compute weighted state and ensure it is the expected shape:
                        weighted_state = (raw_state * feature_weights).flatten()
                        expected_size = agent.state_size
                        if weighted_state.shape[0] != expected_size:
                            weighted_state = np.zeros(expected_size)
                        action = agent.act(weighted_state)
                        sector_return = current_sector_window[sector].iloc[t]
                        portfolio_return = sector_return  # Adjust as needed
                        returns_history = current_sector_window[sector].iloc[:t].values if t > 0 else []
                        reward = self.compute_reward(portfolio_return, returns_history)
                        cumulative_rewards += reward
                        episode_returns.append(portfolio_return)
                        # Store experience using the weighted state
                        for e in range(3):
                            agent.remember(weighted_state, action, reward, weighted_state, done=False)
                            if len(agent.memory) > batch_size:
                                agent.replay(batch_size)
                        sector_allocations.append(action)
                    model_sector_weights.append(sector_allocations)
                normalized_weights = np.array(model_sector_weights)
                sum_weights = normalized_weights.sum(axis=1, keepdims=True)
                sum_weights[sum_weights == 0] = 1
                normalized_weights = normalized_weights / sum_weights
                dynamic_weights_over_time[model_name].append(normalized_weights)

            for t in range(len(current_sector_window)):
                meta_allocations = []
                for sector_index, sector in enumerate(sector_returns.columns.difference(['date'])):
                    weighted_allocation = sum(
                        weights[sector][model_name] * dynamic_weights_over_time[model_name][0][t][sector_index]
                        for model_name in model_names
                    )
                    meta_allocations.append(weighted_allocation)
                meta_model_allocations.append(meta_allocations)

            mean_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            sharpe_ratio = mean_return / (std_return + 1e-10)
            rewards_over_episodes.append(cumulative_rewards)
            sharpe_ratios_over_episodes.append(sharpe_ratio)

        backtest_results = []
        for model_name, weights_list in dynamic_weights_over_time.items():
            model_portfolio_returns = []
            for t in range(len(weights_list[0])):
                if t >= len(current_sector_window):
                    break
                weighted_return = sum(
                    weights_list[0][t][i] * current_sector_window.iloc[t, i]
                    for i in range(len(sectors))
                )
                model_portfolio_returns.append(float(weighted_return))
            model_portfolio_returns = np.array(model_portfolio_returns)
            cumulative_portfolio_return = (1 + model_portfolio_returns).cumprod() - 1
            backtest_results.append({
                'model': model_name,
                'cumulative_returns': cumulative_portfolio_return,
                'returns over time': model_portfolio_returns,
                'weights': dynamic_weights_over_time
            })
        meta_portfolio_returns = []
        for t in range(len(meta_model_allocations)):
            if t >= len(current_sector_window):
                break
            weighted_return = sum(
                meta_model_allocations[t][i] * current_sector_window.iloc[t, i]
                for i in range(len(sectors))
            )
            meta_portfolio_returns.append(float(weighted_return))
        meta_portfolio_returns = np.array(meta_portfolio_returns)
        cumulative_meta_returns = (1 + meta_portfolio_returns).cumprod() - 1
        backtest_results.append({
            'model': 'Meta-Model',
            'cumulative_returns': cumulative_meta_returns,
            'returns over time': meta_portfolio_returns
        })
        final_allocations = {}
        for model_name in model_names:
            final_alloc_vector = dynamic_weights_over_time[model_name][0][-1]
            final_allocations[model_name] = dict(zip(sectors, final_alloc_vector))
        return backtest_results, rewards_over_episodes, sharpe_ratios_over_episodes, final_allocations

    @staticmethod
    def calculate_portfolio_metrics(returns_log, dates) -> dict:
        if not isinstance(returns_log, pd.Series):
            returns_log = pd.Series(returns_log)
        returns = np.exp(returns_log) - 1
        if returns.isnull().all() or len(returns) == 0:
            print("Warning: returns data is empty or NaN.")
            return {
                'Annualized Return': 0,
                'Annualized Volatility': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0,
                'Sortino Ratio': 0
            }
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (12 / len(returns)) - 1
        annualized_volatility = np.std(returns) * np.sqrt(12)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_max - cumulative_returns) / cumulative_max
        drawdown.fillna(0, inplace=True)
        max_drawdown = drawdown.max()
        downside_deviation = np.std([r for r in returns if r < 0]) * np.sqrt(12)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
        return {
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Sortino Ratio': sortino_ratio
        }

    @staticmethod
    def display_metrics(metrics_dict: dict):
        print("\nPortfolio Metrics:")
        for model, metrics in metrics_dict.items():
            print(f"\nModel: {model}")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.5f}")
                else:
                    print(f"{metric}: {value}")


###############################################################################
#                                   MAIN                                      #
###############################################################################
def main():
    # Set TensorFlow threading (optional, depending on system)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)

    # Set up file paths and load necessary datasets
    sector_data = pd.read_csv('aligned_sector_etf_monthly.csv')
    macro_data = pd.read_csv('monthly_macro_indicators_beginning_of_month.csv')
    macro_data.columns = macro_data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    sp500 = pd.read_csv('s&p500_monthly.csv')
    sp500['returns'] = np.log(sp500['^GSPC'] / sp500['^GSPC'].shift(1))
    sp500['returns'] = sp500['returns'].fillna(method='bfill')
    sp500['date'] = pd.to_datetime(sp500['date'])
    sp500 = sp500.drop(['date', '^GSPC'], axis=1)

    # Load RFE and hyperparameter tuning results
    rfe_results = pd.read_excel('rfe_results_cleaned.xlsx')
    rfe_features_per_sector = {
        row['Sector']: row['Selected_Features'].split(', ')
        for _, row in rfe_results.iterrows()
    }
    hyperparameter_tuning_results = pd.read_excel('hyperparameter_tuning_results_cleaned.xlsx')

    # Process the data
    data_processor = DataProcessor()
    sector_returns = data_processor.compute_returns(sector_data)
    macro = data_processor.preprocess_macro(macro_data)
    lags = [3, 6, 9, 12,18]
    lagged_data = data_processor.future_engineering(macro, lags)
    scaler = None  # Assume you create a scaler and standardize the data here
    # For simplicity, we standardize numeric columns (this is an example)
    numeric_cols = lagged_data.columns.difference(['date'])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    lagged_data[numeric_cols] = scaler.fit_transform(lagged_data[numeric_cols])
    roc_data = data_processor.rate_of_change(macro)
    numeric_cols = roc_data.columns.difference(['date'])
    roc_data[numeric_cols] = scaler.fit_transform(roc_data[numeric_cols])
    merged_macro = pd.concat([lagged_data, roc_data], axis=1)

    # Train and evaluate base models
    trainer = ModelTrainer(sector_returns, merged_macro, hyperparameter_tuning_results, rfe_features_per_sector)
    results, model_performance, feature_importance_data, trained_models, futures = trainer.train_and_evaluate_models()

    # Stacking ensemble for each sector
    weights = {}
    ensemble_trainer = EnsembleTrainer()
    for sector in sector_returns.columns.difference(['date']):
        X = merged_macro[futures[sector]]
        y = sector_returns[sector]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=RANDOM_SEED)
        meta_model, meta_predictions, X_stack_test, best_weights, meta_model_metrics = ensemble_trainer.train_stacking_ensemble(
            X_train, y_train, X_test, y_test, trained_models[sector], meta_model_type="RandomForest"
        )
        weights[sector] = best_weights
        model_performance[sector]["Stacked_Model"] = meta_model_metrics

    # Precompute SHAP values for later use
    model_names = ['random_forest', 'lgbm']
    state_size = 10
    action_size = len(sector_returns.columns.difference(['date']))
    # Assume precompute_shap_values is implemented similar to below:
    def precompute_shap_values(trained_models, macro_data, state_size, model_names, features):
        shap_values_cache = {}
        for sector, models in trained_models.items():
            print(f"Precomputing SHAP values for sector: {sector}")
            shap_values_cache[sector] = {}
            X_sample = macro_data[features[sector]].sample(n=min(90, len(macro_data)), random_state=100)
            for model_name, model in zip(model_names, models):
                if isinstance(model, (RandomForestRegressor, LGBMRegressor, XGBRegressor)):
                    explainer = shap.TreeExplainer(model)
                elif isinstance(model, ElasticNet):
                    explainer = shap.LinearExplainer(model, X_sample)
                elif isinstance(model, (SVR, MLPRegressor)):
                    explainer = shap.KernelExplainer(model.predict, X_sample)
                else:
                    continue
                shap_values = explainer.shap_values(X_sample)
                mean_shap = shap_values.mean(axis=0)  # keep the sign
                top_indices = np.argsort(-np.abs(mean_shap))[:state_size]
                top_features = [features[sector][i] for i in top_indices]
                top_weights = mean_shap[top_indices]
                norm_weights = top_weights / np.sum(np.abs(top_weights))
                shap_values_cache[sector][model_name] = {"features": top_features, "weights": norm_weights}
        return shap_values_cache

    shap_values_cache = precompute_shap_values(trained_models, merged_macro, state_size, model_names, futures)

    # Run backtest
    backtester = Backtester()
    # Run backtest and get final allocations
    backtest_results, rewards, sharpe_ratios, final_allocations = backtester.vectorized_backtest(
        sector_returns, merged_macro, trained_models, futures, model_names,
        action_size, weights, shap_values_cache, window_size=302, step=4
    )

    # Print final allocations for each model
    print("Final allocations for each base model:")
    for model_name, alloc_dict in final_allocations.items():
        print(f"Model: {model_name}")
        for sector, alloc in alloc_dict.items():
            print(f"  {sector}: {alloc:.4f}")
        print()

    # Calculate portfolio metrics (example using S&P500 and models)
    model_metrics = {}
    for result in backtest_results:
        returns = result['returns over time']
        metrics = backtester.calculate_portfolio_metrics(returns, sector_returns['date'])
        model_metrics[result['model']] = metrics
    sp500_metrics = backtester.calculate_portfolio_metrics(sp500['returns'], sp500.index)
    model_metrics['S&P500'] = sp500_metrics

    backtest_results.append({
        'model': 'S&P500',
        'cumulative_returns': (1 + sp500['returns']).cumprod() - 1,
        'returns over time': sp500['returns'].tolist()
    })

    Backtester.display_metrics(model_metrics)

    # Plot results
    Plotter.plot_cumulative_returns(backtest_results, sector_returns['date'])
    Plotter.plot_volatility_vs_return(backtest_results)
    Plotter.plot_drawdown_fixed(backtest_results, sector_returns['date'])
    # You can also call interactive Plotly functions here

if __name__ == "__main__":
    main()
