"""
Feature Engineering Module for AI Trading Assistant
Handles technical indicator calculations and data preprocessing
"""

import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import logging
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculates various technical indicators for trading analysis"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus technical indicators
        """
        try:
            logger.info("Calculating technical indicators")
            
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return df
            
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # Calculate trend indicators
            result_df = self._calculate_trend_indicators(result_df)
            
            # Calculate momentum indicators
            result_df = self._calculate_momentum_indicators(result_df)
            
            # Calculate volatility indicators
            result_df = self._calculate_volatility_indicators(result_df)
            
            # Calculate volume indicators
            result_df = self._calculate_volume_indicators(result_df)
            
            # Calculate support/resistance levels
            result_df = self._calculate_support_resistance(result_df)
            
            # Remove any rows with NaN values (from indicator calculations)
            result_df = result_df.dropna()
            
            logger.info(f"Calculated {len(result_df.columns) - 5} technical indicators")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        try:
            # Simple Moving Averages
            df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            # Exponential Moving Averages
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            
            # MACD
            df['MACD'] = ta.trend.macd_diff(df['Close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
            df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
            
            # Parabolic SAR
            df['PSAR'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
            
            # ADX (Average Directional Index)
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
            
            # Golden Cross and Death Cross signals
            df['Golden_Cross'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
            df['Death_Cross'] = np.where(df['SMA_50'] < df['SMA_200'], 1, 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        try:
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'])
            
            # Stochastic Oscillator
            df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            
            # Williams %R
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # ROC (Rate of Change)
            df['ROC'] = ta.momentum.roc(df['Close'])
            
            # CCI (Commodity Channel Index)
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            # Money Flow Index
            df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = bb.bollinger_wband()
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR (Average True Range)
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Keltner Channel
            kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            df['KC_Upper'] = kc.keltner_channel_hband()
            df['KC_Middle'] = kc.keltner_channel_mband()
            df['KC_Lower'] = kc.keltner_channel_lband()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        try:
            # Volume SMA
            df['Volume_SMA_20'] = ta.volume.volume_sma(df['Close'], df['Volume'])
            
            # On Balance Volume
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            # Volume Rate of Change
            df['VROC'] = ta.volume.volume_roc(df['Volume'])
            
            # Chaikin Money Flow
            df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Volume Weighted Average Price
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels"""
        try:
            # Pivot Points
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low']
            df['S1'] = 2 * df['Pivot'] - df['High']
            df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
            df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
            
            # Price position relative to support/resistance
            df['Price_vs_Pivot'] = (df['Close'] - df['Pivot']) / df['Pivot']
            df['Price_vs_R1'] = (df['Close'] - df['R1']) / df['R1']
            df['Price_vs_S1'] = (df['Close'] - df['S1']) / df['S1']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return df

class DataPreprocessor:
    """Handles data preprocessing for machine learning models"""
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = None
        self.target_columns = None
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Close', 
                        lookback_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for time series prediction
        
        Args:
            df: DataFrame with technical indicators
            target_col: Column to predict
            lookback_days: Number of days to look back for features
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        try:
            logger.info(f"Preparing features with {lookback_days} day lookback")
            
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return np.array([]), np.array([])
            
            # Select feature columns (exclude date and target)
            exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Add target column to features for lookback
            all_features = feature_cols + [target_col]
            
            # Create sequences
            features, targets = self._create_sequences(df[all_features], lookback_days)
            
            logger.info(f"Created {len(features)} feature sequences with {features.shape[1]} features each")
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return np.array([]), np.array([])
    
    def _create_sequences(self, data: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series sequences for LSTM"""
        try:
            features = []
            targets = []
            
            for i in range(lookback, len(data)):
                # Features: lookback days of data
                feature_seq = data.iloc[i-lookback:i].values
                features.append(feature_seq)
                
                # Target: next day's close price
                target = data.iloc[i][-1]  # Last column is the target
                targets.append(target)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            return np.array([]), np.array([])
    
    def scale_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using the specified scaler
        
        Args:
            features: Feature array to scale
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Scaled features
        """
        try:
            if self.scaler is None or fit:
                if self.scaler_type == 'minmax':
                    self.scaler = MinMaxScaler()
                else:
                    self.scaler = StandardScaler()
                
                # Reshape for scaler (flatten the lookback dimension)
                original_shape = features.shape
                features_reshaped = features.reshape(-1, features.shape[-1])
                
                if fit:
                    features_scaled = self.scaler.fit_transform(features_reshaped)
                else:
                    features_scaled = self.scaler.transform(features_reshaped)
                
                # Reshape back to original shape
                return features_scaled.reshape(original_shape)
            else:
                # Use existing scaler
                original_shape = features.shape
                features_reshaped = features.reshape(-1, features.shape[-1])
                features_scaled = self.scaler.transform(features_reshaped)
                return features_scaled.reshape(original_shape)
                
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return features
    
    def inverse_scale_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets back to original scale"""
        try:
            if self.scaler is not None:
                return self.scaler.inverse_transform(scaled_targets.reshape(-1, 1)).flatten()
            return scaled_targets
        except Exception as e:
            logger.error(f"Error inverse scaling targets: {str(e)}")
            return scaled_targets

class FeatureSelector:
    """Selects the most important features for the model"""
    
    def __init__(self, method: str = 'correlation'):
        """
        Initialize feature selector
        
        Args:
            method: Selection method ('correlation', 'pca', 'variance')
        """
        self.method = method
        self.selected_features = None
        
    def select_features(self, df: pd.DataFrame, target_col: str = 'Close', 
                       n_features: int = 20) -> List[str]:
        """
        Select the most important features
        
        Args:
            df: DataFrame with features
            target_col: Target column
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            logger.info(f"Selecting {n_features} features using {self.method} method")
            
            if self.method == 'correlation':
                return self._correlation_selection(df, target_col, n_features)
            elif self.method == 'pca':
                return self._pca_selection(df, target_col, n_features)
            elif self.method == 'variance':
                return self._variance_selection(df, target_col, n_features)
            else:
                logger.warning(f"Unknown method {self.method}, using correlation")
                return self._correlation_selection(df, target_col, n_features)
                
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            return []
    
    def _correlation_selection(self, df: pd.DataFrame, target_col: str, n_features: int) -> List[str]:
        """Select features based on correlation with target"""
        try:
            # Calculate correlations
            correlations = df.corr()[target_col].abs().sort_values(ascending=False)
            
            # Remove target column and select top features
            correlations = correlations[correlations.index != target_col]
            selected = correlations.head(n_features).index.tolist()
            
            logger.info(f"Selected features based on correlation: {selected[:5]}...")
            return selected
            
        except Exception as e:
            logger.error(f"Error in correlation selection: {str(e)}")
            return []
    
    def _pca_selection(self, df: pd.DataFrame, target_col: str, n_features: int) -> List[str]:
        """Select features using PCA"""
        try:
            # Remove target and non-numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            # Apply PCA
            pca = PCA(n_components=min(n_features, len(numeric_cols)))
            pca.fit(df[numeric_cols])
            
            # Get feature importance scores
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=numeric_cols
            )
            
            # Select features with highest absolute values in first few PCs
            importance_scores = feature_importance.abs().sum(axis=1).sort_values(ascending=False)
            selected = importance_scores.head(n_features).index.tolist()
            
            logger.info(f"Selected features using PCA: {selected[:5]}...")
            return selected
            
        except Exception as e:
            logger.error(f"Error in PCA selection: {str(e)}")
            return []
    
    def _variance_selection(self, df: pd.DataFrame, target_col: str, n_features: int) -> List[str]:
        """Select features based on variance"""
        try:
            # Remove target column
            feature_cols = [col for col in df.columns if col != target_col]
            
            # Calculate variance for each feature
            variances = df[feature_cols].var().sort_values(ascending=False)
            selected = variances.head(n_features).index.tolist()
            
            logger.info(f"Selected features based on variance: {selected[:5]}...")
            return selected
            
        except Exception as e:
            logger.error(f"Error in variance selection: {str(e)}")
            return []

# Example usage and testing
if __name__ == "__main__":
    # Test technical indicators
    from data_collector import MarketDataCollector
    
    # Get sample data
    collector = MarketDataCollector()
    aapl_data = collector.get_stock_data('AAPL', period='6mo')
    
    if not aapl_data.empty:
        # Calculate indicators
        ti = TechnicalIndicators()
        aapl_with_indicators = ti.calculate_all_indicators(aapl_data)
        
        print(f"Original columns: {len(aapl_data.columns)}")
        print(f"With indicators: {len(aapl_with_indicators.columns)}")
        print(f"New columns: {[col for col in aapl_with_indicators.columns if col not in aapl_data.columns][:10]}")
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        features, targets = preprocessor.prepare_features(aapl_with_indicators, lookback_days=20)
        
        if len(features) > 0:
            print(f"Features shape: {features.shape}")
            print(f"Targets shape: {targets.shape}")
            
            # Test scaling
            scaled_features = preprocessor.scale_features(features, fit=True)
            print(f"Scaled features shape: {scaled_features.shape}")
        
        # Test feature selection
        selector = FeatureSelector(method='correlation')
        selected_features = selector.select_features(aapl_with_indicators, n_features=15)
        print(f"Selected features: {selected_features}")
