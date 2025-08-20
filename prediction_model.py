"""
Prediction Model Module for AI Trading Assistant
Implements LSTM neural network for time series price prediction
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Union
import os
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMPredictor:
    """LSTM-based time series predictor for stock prices"""
    
    def __init__(self, lookback_days: int = 30, prediction_days: int = 1):
        """
        Initialize LSTM predictor
        
        Args:
            lookback_days: Number of days to look back for features
            prediction_days: Number of days to predict ahead
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.model = None
        self.history = None
        self.is_trained = False
        self.feature_columns = None
        self.target_column = None
        
        # Model hyperparameters
        self.lstm_units = [128, 64, 32]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        self.validation_split = 0.2
        
        # Performance metrics
        self.training_metrics = {}
        self.test_metrics = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build the LSTM model architecture
        
        Args:
            input_shape: Shape of input data (lookback_days, n_features)
            
        Returns:
            Compiled Keras model
        """
        try:
            logger.info(f"Building LSTM model with input shape: {input_shape}")
            
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                units=self.lstm_units[0],
                return_sequences=True,
                input_shape=input_shape
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
            
            # Additional LSTM layers
            for i in range(1, len(self.lstm_units)):
                model.add(LSTM(
                    units=self.lstm_units[i],
                    return_sequences=(i < len(self.lstm_units) - 1)
                ))
                model.add(BatchNormalization())
                model.add(Dropout(self.dropout_rate))
            
            # Dense layers for output
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(self.prediction_days, activation='linear'))
            
            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("LSTM model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (features, targets) arrays
        """
        try:
            logger.info(f"Preparing data with {self.lookback_days} day lookback")
            
            if df.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Select feature columns (exclude date and target)
            exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]
            self.target_column = target_col
            
            # Add target column to features for lookback
            all_features = self.feature_columns + [target_col]
            
            # Create sequences
            features, targets = self._create_sequences(df[all_features])
            
            logger.info(f"Created {len(features)} feature sequences with {features.shape[1]} features each")
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series sequences for LSTM"""
        try:
            features = []
            targets = []
            
            for i in range(self.lookback_days, len(data) - self.prediction_days + 1):
                # Features: lookback days of data
                feature_seq = data.iloc[i-self.lookback_days:i].values
                features.append(feature_seq)
                
                # Target: next prediction_days values
                target_seq = data.iloc[i:i+self.prediction_days][-1].values  # Last column is target
                targets.append(target_seq)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def train(self, features: np.ndarray, targets: np.ndarray, 
              validation_data: Optional[Tuple] = None) -> Dict:
        """
        Train the LSTM model
        
        Args:
            features: Training features
            targets: Training targets
            validation_data: Optional validation data tuple
            
        Returns:
            Training history dictionary
        """
        try:
            logger.info("Starting LSTM model training")
            
            if self.model is None:
                self.model = self.build_model((features.shape[1], features.shape[2]))
            
            # Callbacks for better training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    'best_lstm_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            self.history = self.model.fit(
                features,
                targets,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            logger.info("LSTM model training completed")
            
            # Calculate training metrics
            self._calculate_training_metrics(features, targets)
            
            return self.history.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def _calculate_training_metrics(self, features: np.ndarray, targets: np.ndarray):
        """Calculate training performance metrics"""
        try:
            # Predictions on training data
            train_predictions = self.model.predict(features)
            
            # Calculate metrics
            mse = mean_squared_error(targets, train_predictions)
            mae = mean_absolute_error(targets, train_predictions)
            r2 = r2_score(targets, train_predictions)
            
            self.training_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            logger.info(f"Training metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating training metrics: {str(e)}")
    
    def evaluate(self, test_features: np.ndarray, test_targets: np.ndarray) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_features: Test features
            test_targets: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            logger.info("Evaluating model on test data")
            
            # Make predictions
            test_predictions = self.model.predict(test_features)
            
            # Calculate metrics
            mse = mean_squared_error(test_targets, test_predictions)
            mae = mean_absolute_error(test_targets, test_predictions)
            r2 = r2_score(test_targets, test_predictions)
            
            self.test_metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'predictions': test_predictions,
                'actuals': test_targets
            }
            
            logger.info(f"Test metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
            
            return self.test_metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            features: Input features for prediction
            
        Returns:
            Predicted values
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(features)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_future(self, last_sequence: np.ndarray, days_ahead: int = 5) -> np.ndarray:
        """
        Predict future values using recursive prediction
        
        Args:
            last_sequence: Last known sequence of data
            days_ahead: Number of days to predict ahead
            
        Returns:
            Array of predicted values
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                # Make prediction for next day
                next_pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape))
                predictions.append(next_pred[0])
                
                # Update sequence for next prediction (shift and add prediction)
                # This is a simplified approach - in practice you might want more sophisticated handling
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_pred[0]
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error making future predictions: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            if self.is_trained:
                self.model.save(filepath)
                logger.info(f"Model saved to {filepath}")
            else:
                logger.warning("Cannot save untrained model")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            self.model = load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        try:
            if self.history is None:
                logger.warning("No training history available")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            ax1.plot(self.history.history['loss'], label='Training Loss')
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # MAE plot
            ax2.plot(self.history.history['mae'], label='Training MAE')
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
    
    def plot_predictions(self, save_path: Optional[str] = None):
        """Plot actual vs predicted values"""
        try:
            if not self.test_metrics or 'predictions' not in self.test_metrics:
                logger.warning("No test predictions available")
                return
            
            predictions = self.test_metrics['predictions'].flatten()
            actuals = self.test_metrics['actuals'].flatten()
            
            plt.figure(figsize=(12, 6))
            plt.plot(actuals, label='Actual', alpha=0.7)
            plt.plot(predictions, label='Predicted', alpha=0.7)
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Predictions plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")

class ModelEnsemble:
    """Ensemble of multiple prediction models for better accuracy"""
    
    def __init__(self, models: List[LSTMPredictor], weights: Optional[List[float]] = None):
        """
        Initialize ensemble
        
        Args:
            models: List of trained LSTM models
            weights: Optional weights for each model (will be equal if not provided)
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        try:
            predictions = []
            
            for model in self.models:
                pred = model.predict(features)
                predictions.append(pred)
            
            # Weighted average of predictions
            weighted_pred = np.zeros_like(predictions[0])
            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                weighted_pred += weight * pred
            
            return weighted_pred
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Test LSTM predictor
    from data_collector import MarketDataCollector
    from feature_engineering import TechnicalIndicators, DataPreprocessor
    
    # Get sample data
    collector = MarketDataCollector()
    aapl_data = collector.get_stock_data('AAPL', period='1y')
    
    if not aapl_data.empty:
        # Calculate technical indicators
        ti = TechnicalIndicators()
        aapl_with_indicators = ti.calculate_all_indicators(aapl_data)
        
        # Prepare data
        preprocessor = DataPreprocessor()
        features, targets = preprocessor.prepare_features(aapl_with_indicators, lookback_days=30)
        
        if len(features) > 0:
            # Scale features
            scaled_features = preprocessor.scale_features(features, fit=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_features, targets, test_size=0.2, random_state=42
            )
            
            # Train LSTM model
            lstm = LSTMPredictor(lookback_days=30)
            
            try:
                # Train model
                history = lstm.train(X_train, y_train)
                
                # Evaluate model
                test_metrics = lstm.evaluate(X_test, y_test)
                
                print(f"Training completed successfully!")
                print(f"Test R² Score: {test_metrics['r2']:.4f}")
                print(f"Test RMSE: {test_metrics['rmse']:.4f}")
                
                # Make predictions
                predictions = lstm.predict(X_test[:5])
                print(f"Sample predictions: {predictions.flatten()}")
                print(f"Actual values: {y_test[:5].flatten()}")
                
                # Plot results
                lstm.plot_training_history()
                lstm.plot_predictions()
                
            except Exception as e:
                print(f"Training failed: {str(e)}")
                print("This might be due to insufficient data or memory constraints")
