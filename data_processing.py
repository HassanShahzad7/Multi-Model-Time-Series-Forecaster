import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
from datetime import datetime
import talib
warnings.filterwarnings('ignore')

class MoATDataPreprocessor:
    def __init__(self, csv_file, look_back=20, predict_ahead=1, patch_len=5, stride=2):
        self.csv_file = csv_file
        self.look_back = look_back
        self.predict_ahead = predict_ahead
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches
        self.num_patches = (look_back - patch_len) // stride + 1
        
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Different scalers for different feature types
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.technical_scaler = StandardScaler()
        
        # Date splits
        self.train_end_date = '2024-06-10'
        self.test_start_date = '2024-06-11'
        
        print(f"📊 Enhanced MoAT Preprocessor:")
        print(f"   • Look-back: {self.look_back}")
        print(f"   • Patch length: {self.patch_len}")
        print(f"   • Stride: {self.stride}")
        print(f"   • Number of patches: {self.num_patches}")
        print(f"   • Train data: 2022-01-01 to {self.train_end_date}")
        print(f"   • Test data: {self.test_start_date} to 2024-12-31")
    
    def load_and_clean_data(self):
        """Load and clean data with date filtering"""
        print("\n🚀 Enhanced MoAT Preprocessing Pipeline\n")
        
        df = pd.read_csv(self.csv_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter date range (2022 onwards)
        df = df[df['date'] >= '2022-01-01']
        df = df[df['date'] <= '2024-12-31']
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove unnecessary columns
        cols_to_drop = ['url', 'dividends', 'stock_splits', 'sentiment_label']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Remove weekends
        df = df[df['date'].dt.weekday < 5]
        
        print(f"   • Total samples: {len(df)}")
        
        return df
    
    def create_advanced_features(self, df):
        """Create advanced technical features as per MoAT paper"""
        print("📈 Creating advanced features...")
        
        # Convert to float64 for TA-Lib
        df['open'] = df['open'].astype(np.float64)
        df['high'] = df['high'].astype(np.float64)
        df['low'] = df['low'].astype(np.float64)
        df['close'] = df['close'].astype(np.float64)
        df['volume'] = df['volume'].astype(np.float64)
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low']
        
        # Volume features
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_trend'] = df['volume_ma5'] / df['volume_ma20']
        
        # Technical indicators using TA-Lib
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Moving averages
        df['sma_5'] = talib.SMA(close, timeperiod=5)
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Price position relative to MAs
        df['price_sma5_ratio'] = df['close'] / df['sma_5']
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        df['rsi_scaled'] = df['rsi'] / 100.0
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_norm'] = df['macd'] / df['close']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ATR (volatility)
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['atr_norm'] = df['atr'] / df['close']
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        df['stoch_k'] = slowk / 100.0
        df['stoch_d'] = slowd / 100.0
        
        # OBV (On Balance Volume)
        df['obv'] = talib.OBV(close, volume)
        df['obv_ma'] = df['obv'].rolling(window=20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_ma']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def seasonal_trend_decomposition(self, data, period=5):
        """
        Perform seasonal-trend decomposition using moving average
        Following the MoAT paper approach
        """
        # Moving average for trend
        kernel = np.ones(period) / period
        trend = np.convolve(data, kernel, mode='same')
        
        # Handle edges
        half_period = period // 2
        trend[:half_period] = data[:half_period]
        trend[-half_period:] = data[-half_period:]
        
        # Seasonal is residual
        seasonal = data - trend
        
        return trend, seasonal
    
    def create_patches(self, data):
        """Create patches from sequence data"""
        patches = []
        for i in range(0, self.look_back - self.patch_len + 1, self.stride):
            patch = data[i:i+self.patch_len]
            patches.append(patch)
        return np.array(patches)
    
    def encode_text_patches_with_attention(self, texts, patch_indices):
        """
        Encode text with attention mechanism as per MoAT paper
        Each patch gets weighted text embeddings
        """
        text_patches = []
        
        for patch_idx in range(len(patch_indices)):
            start_idx = patch_idx * self.stride
            end_idx = min(start_idx + self.patch_len, len(texts))
            
            patch_texts = []
            for i in range(start_idx, end_idx):
                if i < len(texts) and pd.notna(texts[i]) and texts[i] != 'No significant news':
                    patch_texts.append(texts[i])
            
            if patch_texts:
                # Encode all texts in patch
                embeddings = self.text_encoder.encode(patch_texts)
                
                # Simple attention mechanism (can be learned in actual implementation)
                # Weight by recency (more recent = higher weight)
                weights = np.linspace(0.5, 1.0, len(embeddings))
                weights = weights / weights.sum()
                
                # Weighted average
                patch_embedding = np.average(embeddings, axis=0, weights=weights)
            else:
                patch_embedding = np.zeros(768)
            
            text_patches.append(patch_embedding)
        
        return np.array(text_patches)
    
    def prepare_sequences(self, df, is_train=True):
        """Prepare sequences with trend-seasonal decomposition"""
        # Select features
        price_features = ['open', 'high', 'low', 'close']
        volume_features = ['volume', 'volume_ratio', 'volume_trend']
        technical_features = [
            'returns', 'log_returns', 'price_range', 'close_to_high', 'close_to_low',
            'price_sma5_ratio', 'price_sma20_ratio', 'rsi_scaled', 
            'macd_norm', 'bb_position', 'bb_width', 'atr_norm',
            'stoch_k', 'stoch_d', 'obv_ratio'
        ]
        
        # Separate feature groups for different scaling
        price_data = df[price_features].values
        volume_data = df[volume_features].values
        technical_data = df[technical_features].values
        
        # Scale features
        if is_train:
            price_scaled = self.price_scaler.fit_transform(price_data)
            volume_scaled = self.volume_scaler.fit_transform(volume_data)
            technical_scaled = self.technical_scaler.fit_transform(technical_data)
        else:
            price_scaled = self.price_scaler.transform(price_data)
            volume_scaled = self.volume_scaler.transform(volume_data)
            technical_scaled = self.technical_scaler.transform(technical_data)
        
        # Combine scaled features
        feature_data = np.concatenate([price_scaled, volume_scaled, technical_scaled], axis=1)
        
        sequences = []
        text_sequences = []
        trend_sequences = []
        seasonal_sequences = []
        targets = []
        dates = []
        
        texts = df['text'].tolist()
        
        for i in range(self.look_back, len(df) - self.predict_ahead + 1):
            # Time series sequence
            seq = feature_data[i-self.look_back:i]
            
            # Apply trend-seasonal decomposition to each feature
            trend_data = []
            seasonal_data = []
            
            for feat_idx in range(seq.shape[1]):
                trend, seasonal = self.seasonal_trend_decomposition(seq[:, feat_idx])
                trend_data.append(trend)
                seasonal_data.append(seasonal)
            
            trend_data = np.array(trend_data).T
            seasonal_data = np.array(seasonal_data).T
            
            # Create patches
            seq_patches = self.create_patches(seq)
            trend_patches = self.create_patches(trend_data)
            seasonal_patches = self.create_patches(seasonal_data)
            
            # Text patches with attention
            text_seq = texts[i-self.look_back:i]
            patch_indices = list(range(self.num_patches))
            text_patches = self.encode_text_patches_with_attention(text_seq, patch_indices)
            
            # Target (next day's return)
            target = df['returns'].iloc[i:i+self.predict_ahead].values[0]
            
            sequences.append(seq_patches)
            text_sequences.append(text_patches)
            trend_sequences.append(trend_patches)
            seasonal_sequences.append(seasonal_patches)
            targets.append(target)
            dates.append(df['date'].iloc[i])
        
        return (np.array(sequences), np.array(text_sequences), 
                np.array(trend_sequences), np.array(seasonal_sequences),
                np.array(targets), dates)
    
    def preprocess(self):
        """Complete preprocessing pipeline"""
        # Load and clean data
        df = self.load_and_clean_data()
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Split by date
        train_df = df[df['date'] <= self.train_end_date].copy()
        test_df = df[df['date'] >= self.test_start_date].copy()
        
        print(f"\n📊 Data split:")
        print(f"   • Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
        print(f"   • Test: {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
        
        # Prepare sequences
        print("\n🔄 Preparing sequences...")
        
        (train_seq, train_text, train_trend, train_seasonal, 
         train_targets, train_dates) = self.prepare_sequences(train_df, is_train=True)
        
        (test_seq, test_text, test_trend, test_seasonal,
         test_targets, test_dates) = self.prepare_sequences(test_df, is_train=False)
        
        print(f"\n📊 Sequence shapes:")
        print(f"   • Train sequences: {train_seq.shape}")
        print(f"   • Test sequences: {test_seq.shape}")
        print(f"   • Features per sample: {train_seq.shape[-1]}")
        
        # Create data dictionary
        data_dict = {
            'train': {
                'time_series': train_seq,
                'text': train_text,
                'trend': train_trend,
                'seasonal': train_seasonal,
                'targets': train_targets,
                'dates': train_dates
            },
            'test': {
                'time_series': test_seq,
                'text': test_text,
                'trend': test_trend,
                'seasonal': test_seasonal,
                'targets': test_targets,
                'dates': test_dates
            },
            'feature_dim': train_seq.shape[-1],
            'num_patches': self.num_patches,
            'patch_len': self.patch_len,
            'stride': self.stride
        }
        
        # Save preprocessed data
        np.savez('moat_preprocessed_data.npz', **data_dict)
        
        # Save scalers
        with open('moat_scalers.pkl', 'wb') as f:
            pickle.dump({
                'price_scaler': self.price_scaler,
                'volume_scaler': self.volume_scaler,
                'technical_scaler': self.technical_scaler
            }, f)
        
        print("\n✅ Enhanced preprocessing complete!")
        print(f"   • Data saved to: moat_preprocessed_data.npz")
        print(f"   • Scalers saved to: moat_scalers.pkl")
        
        # Print feature statistics
        print(f"\n📊 Target statistics:")
        print(f"   • Train - Mean: {np.mean(train_targets):.6f}, Std: {np.std(train_targets):.6f}")
        print(f"   • Test - Mean: {np.mean(test_targets):.6f}, Std: {np.std(test_targets):.6f}")
        
        return data_dict

if __name__ == "__main__":
    preprocessor = MoATDataPreprocessor(
        csv_file='nvda_combined_dataset_2022_2025.csv',
        look_back=20,  # Increased for better context
        predict_ahead=1,
        patch_len=5,   # Adjusted patch length
        stride=2
    )
    data = preprocessor.preprocess()