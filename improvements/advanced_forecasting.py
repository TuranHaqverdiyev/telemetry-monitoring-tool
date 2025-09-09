"""
Advanced multivariate forecasting system for telemetry data.

This module provides:
- Multivariate time series forecasting with cross-channel dependencies  
- LSTM/neural network based predictions
- Ensemble forecasting methods
- Automatic model selection and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import logging
from collections import deque
import warnings

# Machine Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Statistical forecasting
try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Advanced ML forecasting
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ForecastResult:
    """Enhanced forecast result with metadata and confidence intervals."""
    horizon: int
    forecast_values: Dict[str, List[float]]  # channel -> predicted values
    confidence_intervals: Dict[str, List[Tuple[float, float]]]  # channel -> (lower, upper) bounds
    model_type: str
    accuracy_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None  # channel -> feature -> importance
    breach_predictions: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)  # channel -> (threshold, eta_sec)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """Model performance metrics for comparison."""
    model_name: str
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    training_time_seconds: float
    prediction_time_ms: float
    data_requirements: int  # Minimum data points needed
    memory_usage_mb: float

class ForecastModel(ABC):
    """Abstract base class for forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.training_time = 0.0
        self.last_performance: Optional[ModelPerformance] = None
    
    @abstractmethod
    def fit(self, data: Dict[str, List[float]], timestamps: List[float]) -> bool:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate forecasts with confidence intervals."""
        pass
    
    @abstractmethod
    def get_minimum_data_points(self) -> int:
        """Return minimum data points required for training."""
        pass
    
    def evaluate_performance(self, test_data: Dict[str, List[float]], 
                           test_timestamps: List[float]) -> ModelPerformance:
        """Evaluate model performance on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        start_time = time.time()
        predictions = self.predict(len(test_timestamps))
        prediction_time = (time.time() - start_time) * 1000
        
        # Calculate metrics across all channels
        mae_values = []
        mse_values = []
        mape_values = []
        
        for channel, actual_values in test_data.items():
            if channel in predictions.forecast_values:
                predicted_values = predictions.forecast_values[channel]
                min_len = min(len(actual_values), len(predicted_values))
                
                actual = np.array(actual_values[:min_len])
                predicted = np.array(predicted_values[:min_len])
                
                mae_values.append(np.mean(np.abs(actual - predicted)))
                mse_values.append(np.mean((actual - predicted) ** 2))
                
                # MAPE calculation with handling for zero values
                non_zero_mask = actual != 0
                if np.any(non_zero_mask):
                    mape_values.append(np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100)
        
        self.last_performance = ModelPerformance(
            model_name=self.name,
            mae=np.mean(mae_values) if mae_values else float('inf'),
            mse=np.mean(mse_values) if mse_values else float('inf'),
            mape=np.mean(mape_values) if mape_values else float('inf'),
            training_time_seconds=self.training_time,
            prediction_time_ms=prediction_time,
            data_requirements=self.get_minimum_data_points(),
            memory_usage_mb=self._estimate_memory_usage()
        )
        
        return self.last_performance
    
    def _estimate_memory_usage(self) -> float:
        """Estimate model memory usage in MB."""
        # Default implementation - subclasses should override for accuracy
        return 1.0

class UnivariateHoltWintersModel(ForecastModel):
    """Enhanced Holt-Winters exponential smoothing with confidence intervals."""
    
    def __init__(self, seasonal_periods: Optional[int] = None, trend: str = "add", seasonal: str = "add"):
        super().__init__("HoltWinters")
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
    
    def fit(self, data: Dict[str, List[float]], timestamps: List[float]) -> bool:
        """Fit Holt-Winters model for each channel independently."""
        start_time = time.time()
        
        try:
            for channel, values in data.items():
                if len(values) < self.get_minimum_data_points():
                    logger.warning(f"⚠️ Insufficient data for {channel}: {len(values)} < {self.get_minimum_data_points()}")
                    continue
                
                # Scale data
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
                self.scalers[channel] = scaler
                
                # Determine if seasonal modeling is appropriate
                use_seasonal = (self.seasonal_periods and 
                              len(values) >= 2 * self.seasonal_periods and
                              self.seasonal)
                
                try:
                    if STATSMODELS_AVAILABLE:
                        model = ExponentialSmoothing(
                            scaled_values,
                            trend=self.trend,
                            seasonal=self.seasonal if use_seasonal else None,
                            seasonal_periods=self.seasonal_periods if use_seasonal else None,
                            initialization_method="estimated"
                        )
                        fitted_model = model.fit(optimized=True)
                        self.models[channel] = fitted_model
                    else:
                        logger.error("❌ statsmodels not available for Holt-Winters forecasting")
                        return False
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fit seasonal model for {channel}, trying simple trend: {e}")
                    try:
                        # Fallback to simple trend model
                        model = ExponentialSmoothing(
                            scaled_values,
                            trend="add",
                            seasonal=None,
                            initialization_method="estimated"
                        )
                        fitted_model = model.fit(optimized=True)
                        self.models[channel] = fitted_model
                    except Exception as e2:
                        logger.error(f"❌ Failed to fit any model for {channel}: {e2}")
                        continue
            
            self.training_time = time.time() - start_time
            self.is_fitted = len(self.models) > 0
            
            logger.info(f"✅ Fitted Holt-Winters models for {len(self.models)} channels in {self.training_time:.2f}s")
            return self.is_fitted
            
        except Exception as e:
            logger.error(f"❌ Failed to fit Holt-Winters models: {e}")
            return False
    
    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate forecasts with confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast_values = {}
        confidence_intervals = {} 
        
        alpha = 1.0 - confidence_level
        
        for channel, model in self.models.items():
            try:
                # Generate point forecasts
                forecast = model.forecast(steps=horizon)
                
                # Generate confidence intervals using forecast errors
                forecast_errors = model.resid
                error_std = np.std(forecast_errors)
                
                # Simple confidence interval estimation
                z_score = 1.96  # 95% confidence interval
                margin_of_error = z_score * error_std
                
                # Unscale forecasts
                scaler = self.scalers[channel]
                unscaled_forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
                
                forecast_values[channel] = unscaled_forecast.tolist()
                confidence_intervals[channel] = [
                    (val - margin_of_error, val + margin_of_error) 
                    for val in unscaled_forecast
                ]
                
            except Exception as e:
                logger.error(f"❌ Failed to generate forecast for {channel}: {e}")
                forecast_values[channel] = [0.0] * horizon
                confidence_intervals[channel] = [(0.0, 0.0)] * horizon
        
        return ForecastResult(
            horizon=horizon,
            forecast_values=forecast_values,
            confidence_intervals=confidence_intervals,
            model_type=self.name,
            accuracy_metrics={"confidence_level": confidence_level}
        )
    
    def get_minimum_data_points(self) -> int:
        base_requirement = 20
        if self.seasonal_periods:
            return max(base_requirement, 3 * self.seasonal_periods)
        return base_requirement

class MultivariateVARModel(ForecastModel):
    """Vector Autoregression model for multivariate forecasting."""
    
    def __init__(self, max_lags: int = 10, ic: str = 'aic'):
        super().__init__("VAR")
        self.max_lags = max_lags
        self.ic = ic  # Information criterion for lag selection
        self.var_model = None
        self.scaler = StandardScaler()
        self.channel_names: List[str] = []
    
    def fit(self, data: Dict[str, List[float]], timestamps: List[float]) -> bool:
        """Fit VAR model to multivariate data."""
        start_time = time.time()
        
        try:
            if not STATSMODELS_AVAILABLE:
                logger.error("❌ statsmodels not available for VAR forecasting")
                return False
            
            # Create DataFrame from channel data
            min_length = min(len(values) for values in data.values())
            if min_length < self.get_minimum_data_points():
                logger.warning(f"⚠️ Insufficient data for VAR: {min_length} < {self.get_minimum_data_points()}")
                return False
            
            # Align all series to same length
            aligned_data = {}
            for channel, values in data.items():
                aligned_data[channel] = values[-min_length:]
            
            df = pd.DataFrame(aligned_data)
            self.channel_names = list(df.columns)
            
            # Scale data
            scaled_data = self.scaler.fit_transform(df)
            scaled_df = pd.DataFrame(scaled_data, columns=self.channel_names)
            
            # Fit VAR model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                var = VAR(scaled_df)
                self.var_model = var.fit(maxlags=self.max_lags, ic=self.ic)
            
            self.training_time = time.time() - start_time
            self.is_fitted = True
            
            logger.info(f"✅ Fitted VAR model with {self.var_model.k_ar} lags in {self.training_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fit VAR model: {e}")
            return False
    
    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate multivariate forecasts."""
        if not self.is_fitted or not self.var_model:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate forecasts
            forecast = self.var_model.forecast(self.var_model.y, steps=horizon)
            
            # Generate confidence intervals
            forecast_errors = self.var_model.resid
            error_cov = np.cov(forecast_errors.T)
            
            # Unscale forecasts
            unscaled_forecast = self.scaler.inverse_transform(forecast)
            
            forecast_values = {}
            confidence_intervals = {}
            feature_importance = {}
            
            for i, channel in enumerate(self.channel_names):
                forecast_values[channel] = unscaled_forecast[:, i].tolist()
                
                # Simple confidence intervals using diagonal of covariance matrix
                error_std = np.sqrt(error_cov[i, i])
                z_score = 1.96  # 95% confidence
                margin = z_score * error_std
                
                confidence_intervals[channel] = [
                    (val - margin, val + margin) 
                    for val in unscaled_forecast[:, i]
                ]
                
                # Feature importance from VAR coefficients
                if hasattr(self.var_model, 'coefs'):
                    coeffs = self.var_model.coefs
                    importance = {}
                    for j, feature in enumerate(self.channel_names):
                        # Sum absolute coefficients across lags
                        importance[feature] = np.sum(np.abs(coeffs[:, i, j]))
                    
                    # Normalize to sum to 1
                    total_importance = sum(importance.values())
                    if total_importance > 0:
                        importance = {k: v/total_importance for k, v in importance.items()}
                    
                    feature_importance[channel] = importance
            
            return ForecastResult(
                horizon=horizon,
                forecast_values=forecast_values,
                confidence_intervals=confidence_intervals,
                model_type=self.name,
                accuracy_metrics={"confidence_level": confidence_level},
                feature_importance=feature_importance,
                metadata={
                    "lags_used": self.var_model.k_ar,
                    "channels": self.channel_names,
                    "cross_channel_dependencies": True
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate VAR forecast: {e}")
            # Return empty forecast
            return ForecastResult(
                horizon=horizon,
                forecast_values={ch: [0.0] * horizon for ch in self.channel_names},
                confidence_intervals={ch: [(0.0, 0.0)] * horizon for ch in self.channel_names},
                model_type=self.name,
                accuracy_metrics={}
            )
    
    def get_minimum_data_points(self) -> int:
        return max(50, self.max_lags * 5)

class LSTMForecaster(ForecastModel):
    """LSTM-based neural network forecasting model."""
    
    def __init__(self, 
                 sequence_length: int = 20,
                 hidden_size: int = 64, 
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32):
        super().__init__("LSTM")
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM forecasting")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.channel_names: List[str] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, data: Dict[str, List[float]], timestamps: List[float]) -> bool:
        """Fit LSTM model to time series data."""
        if not PYTORCH_AVAILABLE:
            logger.error("❌ PyTorch not available for LSTM forecasting")
            return False
        
        start_time = time.time()
        
        try:
            # Prepare data
            min_length = min(len(values) for values in data.values())
            if min_length < self.get_minimum_data_points():
                logger.warning(f"⚠️ Insufficient data for LSTM: {min_length} < {self.get_minimum_data_points()}")
                return False
            
            # Create multivariate dataset
            aligned_data = {}
            for channel, values in data.items():
                aligned_data[channel] = values[-min_length:]
            
            df = pd.DataFrame(aligned_data)
            self.channel_names = list(df.columns)
            
            # Scale data
            scaled_data = self.scaler.fit_transform(df.values)
            
            # Create sequences for LSTM
            X, y = self._create_sequences(scaled_data)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Create model
            input_size = len(self.channel_names)
            self.model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=input_size,
                dropout=self.dropout
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logger.debug(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            self.model.eval()
            self.training_time = time.time() - start_time
            self.is_fitted = True
            
            logger.info(f"✅ Fitted LSTM model with {len(self.channel_names)} channels in {self.training_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fit LSTM model: {e}")
            return False
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate LSTM forecasts."""
        if not self.is_fitted or not self.model:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            self.model.eval()
            forecasts = []
            
            # Use last sequence as starting point
            last_sequence = torch.FloatTensor(self.scaler.transform(
                np.array([self.channel_names[-self.sequence_length:]])
            )).unsqueeze(0).to(self.device)
            
            # Generate forecasts iteratively
            current_sequence = last_sequence
            with torch.no_grad():
                for _ in range(horizon):
                    next_pred = self.model(current_sequence)
                    forecasts.append(next_pred.cpu().numpy()[0])
                    
                    # Update sequence for next prediction
                    current_sequence = torch.cat([
                        current_sequence[:, 1:, :],
                        next_pred.unsqueeze(1)
                    ], dim=1)
            
            # Unscale forecasts
            forecasts_array = np.array(forecasts)
            unscaled_forecasts = self.scaler.inverse_transform(forecasts_array)
            
            # Organize by channel
            forecast_values = {}
            confidence_intervals = {}
            
            for i, channel in enumerate(self.channel_names):
                forecast_values[channel] = unscaled_forecasts[:, i].tolist()
                
                # Simple confidence intervals (could be improved with prediction intervals)
                std_estimate = np.std(unscaled_forecasts[:, i])
                margin = 1.96 * std_estimate
                confidence_intervals[channel] = [
                    (val - margin, val + margin) 
                    for val in unscaled_forecasts[:, i]
                ]
            
            return ForecastResult(
                horizon=horizon,
                forecast_values=forecast_values,
                confidence_intervals=confidence_intervals,
                model_type=self.name,
                accuracy_metrics={"confidence_level": confidence_level},
                metadata={
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "device": str(self.device)
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate LSTM forecast: {e}")
            return ForecastResult(
                horizon=horizon,
                forecast_values={ch: [0.0] * horizon for ch in self.channel_names},
                confidence_intervals={ch: [(0.0, 0.0)] * horizon for ch in self.channel_names},
                model_type=self.name,
                accuracy_metrics={}
            )
    
    def get_minimum_data_points(self) -> int:
        return self.sequence_length * 3 + 50
    
    def _estimate_memory_usage(self) -> float:
        """Estimate LSTM model memory usage."""
        if not self.model:
            return 1.0
        
        # Rough estimate based on model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter

class LSTMNetwork(nn.Module):
    """LSTM network architecture."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)
        
        return output

class EnsembleForecastEngine:
    """Ensemble forecasting engine that combines multiple models."""
    
    def __init__(self):
        self.models: List[ForecastModel] = []
        self.model_weights: Dict[str, float] = {}
        self.performance_history: List[Dict[str, ModelPerformance]] = deque(maxlen=10)
        
    def add_model(self, model: ForecastModel, weight: float = 1.0):
        """Add a forecasting model to the ensemble."""
        self.models.append(model)
        self.model_weights[model.name] = weight
        logger.info(f"➕ Added {model.name} to ensemble with weight {weight}")
    
    def auto_select_models(self, data: Dict[str, List[float]], 
                          timestamps: List[float]) -> List[str]:
        """Automatically select best models based on data characteristics."""
        data_size = min(len(values) for values in data.values())
        num_channels = len(data)
        
        selected_models = []
        
        # Always include Holt-Winters for baseline
        if STATSMODELS_AVAILABLE:
            selected_models.append("HoltWinters")
        
        # Add VAR for multivariate if multiple channels and sufficient data
        if num_channels > 1 and data_size >= 100 and STATSMODELS_AVAILABLE:
            selected_models.append("VAR")
        
        # Add LSTM for complex patterns with lots of data
        if data_size >= 200 and PYTORCH_AVAILABLE:
            selected_models.append("LSTM")
        
        logger.info(f"🎯 Auto-selected models: {selected_models}")
        return selected_models
    
    def fit_all_models(self, data: Dict[str, List[float]], 
                      timestamps: List[float]) -> Dict[str, bool]:
        """Fit all models in the ensemble."""
        results = {}
        
        for model in self.models:
            try:
                logger.info(f"🔄 Fitting {model.name}...")
                success = model.fit(data, timestamps)
                results[model.name] = success
                
                if success:
                    logger.info(f"✅ {model.name} fitted successfully")
                else:
                    logger.warning(f"⚠️ {model.name} failed to fit")
                    
            except Exception as e:
                logger.error(f"❌ Error fitting {model.name}: {e}")
                results[model.name] = False
        
        return results
    
    def generate_ensemble_forecast(self, horizon: int, 
                                 confidence_level: float = 0.95) -> ForecastResult:
        """Generate ensemble forecast by combining predictions from fitted models."""
        fitted_models = [model for model in self.models if model.is_fitted]
        
        if not fitted_models:
            raise ValueError("No fitted models available for forecasting")
        
        # Get predictions from all fitted models
        model_forecasts = {}
        for model in fitted_models:
            try:
                forecast = model.predict(horizon, confidence_level)
                model_forecasts[model.name] = forecast
            except Exception as e:
                logger.error(f"❌ Error getting forecast from {model.name}: {e}")
        
        if not model_forecasts:
            raise ValueError("No successful forecasts generated")
        
        # Combine forecasts using weighted average
        ensemble_forecast = self._combine_forecasts(model_forecasts, horizon)
        
        return ensemble_forecast
    
    def _combine_forecasts(self, model_forecasts: Dict[str, ForecastResult], 
                          horizon: int) -> ForecastResult:
        """Combine multiple forecasts using weighted averaging."""
        
        # Get all channels from all models
        all_channels = set()
        for forecast in model_forecasts.values():
            all_channels.update(forecast.forecast_values.keys())
        
        combined_values = {}
        combined_intervals = {}
        
        for channel in all_channels:
            channel_forecasts = []
            channel_weights = []
            channel_intervals = []
            
            for model_name, forecast in model_forecasts.items():
                if channel in forecast.forecast_values:
                    channel_forecasts.append(forecast.forecast_values[channel])
                    channel_weights.append(self.model_weights.get(model_name, 1.0))
                    if channel in forecast.confidence_intervals:
                        channel_intervals.append(forecast.confidence_intervals[channel])
            
            if channel_forecasts:
                # Weighted average of forecasts
                weighted_forecast = self._weighted_average(channel_forecasts, channel_weights)
                combined_values[channel] = weighted_forecast
                
                # Combine confidence intervals (conservative approach)
                if channel_intervals:
                    combined_lower = []
                    combined_upper = []
                    
                    for i in range(horizon):
                        lower_bounds = [intervals[i][0] for intervals in channel_intervals if i < len(intervals)]
                        upper_bounds = [intervals[i][1] for intervals in channel_intervals if i < len(intervals)]
                        
                        if lower_bounds and upper_bounds:
                            combined_lower.append(min(lower_bounds))
                            combined_upper.append(max(upper_bounds))
                        else:
                            combined_lower.append(weighted_forecast[i] * 0.95)
                            combined_upper.append(weighted_forecast[i] * 1.05)
                    
                    combined_intervals[channel] = list(zip(combined_lower, combined_upper))
                else:
                    # Default confidence intervals
                    combined_intervals[channel] = [
                        (val * 0.95, val * 1.05) for val in weighted_forecast
                    ]
        
        return ForecastResult(
            horizon=horizon,
            forecast_values=combined_values,
            confidence_intervals=combined_intervals,
            model_type="Ensemble",
            accuracy_metrics={
                "models_used": list(model_forecasts.keys()),
                "confidence_level": 0.95
            },
            metadata={
                "ensemble_size": len(model_forecasts),
                "channels": list(all_channels)
            }
        )
    
    def _weighted_average(self, forecasts: List[List[float]], 
                         weights: List[float]) -> List[float]:
        """Calculate weighted average of multiple forecast series."""
        if not forecasts or not weights:
            return []
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(weights)
            total_weight = len(weights)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted average for each time step
        horizon = min(len(f) for f in forecasts)
        weighted_forecast = []
        
        for i in range(horizon):
            weighted_sum = sum(
                forecast[i] * weight 
                for forecast, weight in zip(forecasts, normalized_weights)
            )
            weighted_forecast.append(weighted_sum)
        
        return weighted_forecast

# Example usage
def example_advanced_forecasting():
    """Example of advanced forecasting system usage."""
    
    # Generate sample data
    np.random.seed(42)
    timestamps = list(range(1000))
    
    # Correlated channels with trends and seasonality
    base_trend = np.linspace(0, 10, 1000)
    seasonal = 2 * np.sin(2 * np.pi * np.array(timestamps) / 50)
    
    data = {
        'temperature': (base_trend + seasonal + np.random.normal(0, 0.5, 1000)).tolist(),
        'voltage': (3.3 + base_trend * 0.1 + seasonal * 0.05 + np.random.normal(0, 0.02, 1000)).tolist(),
        'current': (1.0 + base_trend * 0.05 + seasonal * 0.02 + np.random.normal(0, 0.01, 1000)).tolist()
    }
    
    # Create ensemble forecasting engine
    ensemble = EnsembleForecastEngine()
    
    # Add models
    if STATSMODELS_AVAILABLE:
        holt_winters = UnivariateHoltWintersModel(seasonal_periods=50)
        ensemble.add_model(holt_winters, weight=1.0)
        
        var_model = MultivariateVARModel(max_lags=5)
        ensemble.add_model(var_model, weight=1.5)  # Higher weight for multivariate
    
    if PYTORCH_AVAILABLE:
        lstm_model = LSTMForecaster(sequence_length=20, epochs=50)
        ensemble.add_model(lstm_model, weight=1.2)
    
    # Fit all models
    print("🔄 Fitting ensemble models...")
    fit_results = ensemble.fit_all_models(data, timestamps)
    
    for model_name, success in fit_results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {model_name}: {status}")
    
    # Generate ensemble forecast
    try:
        print("\n🔮 Generating ensemble forecast...")
        forecast = ensemble.generate_ensemble_forecast(horizon=50)
        
        print(f"📊 Forecast summary:")
        print(f"  Horizon: {forecast.horizon} steps")
        print(f"  Channels: {list(forecast.forecast_values.keys())}")
        print(f"  Models used: {forecast.accuracy_metrics.get('models_used', [])}")
        
        # Show sample predictions
        for channel, predictions in forecast.forecast_values.items():
            print(f"  {channel}: [{predictions[0]:.3f}, {predictions[1]:.3f}, ..., {predictions[-1]:.3f}]")
            
    except Exception as e:
        print(f"❌ Ensemble forecasting failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_advanced_forecasting()
