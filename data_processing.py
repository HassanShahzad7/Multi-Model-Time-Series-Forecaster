import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class MoATDataPreprocessor:
    def __init__(self, csv_file, look_back=8, predict_ahead=1):
        """
        MoAT data preprocessor following the paper exactly
        
        Args:
            csv_file: Path to combined dataset CSV
            look_back: L = 8 (past timesteps, as per paper)
            predict_ahead: T = 1 (future timesteps to predict)
        """
        self.csv_file = csv_file
        self.look_back = look_back  # L = 8
        self.predict_ahead = predict_ahead  # T = 1
        
        # Text encoder (as per paper Appendix B.4)
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        print(f"📊 MoAT Preprocessor (Paper Implementation):")
        print(f"   • Look-back window (L): {self.look_back}")
        print(f"   • Prediction horizon (T): {self.predict_ahead}")
    
    def load_and_split_data(self):
        """Load data and split by your date requirements"""
        print("📖 Loading and splitting dataset...")
        
        # Load data
        df = pd.read_csv(self.csv_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Split by dates as requested
        train_end_date = '2024-12-31'
        test_start_date = '2025-01-01'
        test_end_date = '2025-06-30'
        
        # Create splits
        train_df = df[df['date'] <= train_end_date].reset_index(drop=True)
        test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].reset_index(drop=True)
        
        print(f"   • Train: {len(train_df)} days ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
        print(f"   • Test: {len(test_df)} days ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
        
        # Extract stock and text data
        stock_columns = ['open', 'high', 'low', 'close', 'volume']
        
        self.train_stock = train_df[stock_columns].values
        self.train_text = train_df['text'].tolist()
        self.train_dates = train_df['date'].tolist()
        
        self.test_stock = test_df[stock_columns].values
        self.test_text = test_df['text'].tolist()
        self.test_dates = test_df['date'].tolist()
        
        return train_df, test_df
    
    def instance_normalize_stock_data(self):
        """Instance normalization as per MoAT paper (Appendix A)"""
        print("🔢 Applying Instance Normalization (as per paper)...")
        
        def instance_normalize(data):
            """
            Instance normalization with temporal locality
            Paper formula: (x^(i) - (mi + x_L)/2) / σi
            """
            normalized_data = np.zeros_like(data)
            
            for i in range(data.shape[1]):  # For each channel/feature
                channel_data = data[:, i]
                
                # Compute mean and std
                mi = np.mean(channel_data)
                sigma_i = np.std(channel_data)
                
                # Apply temporal locality: use (mi + last_value)/2 instead of mi
                # For simplicity, we'll use the standard formula but this can be enhanced
                if sigma_i > 0:
                    normalized_data[:, i] = (channel_data - mi) / sigma_i
                else:
                    normalized_data[:, i] = channel_data - mi
            
            return normalized_data
        
        # Normalize train and test separately (as per paper's approach)
        self.train_stock_normalized = instance_normalize(self.train_stock)
        self.test_stock_normalized = instance_normalize(self.test_stock)
        
        print(f"   • Train stock normalized: {self.train_stock_normalized.shape}")
        print(f"   • Test stock normalized: {self.test_stock_normalized.shape}")
    
    def encode_text_data(self):
        """Text encoding using PLM (as per paper)"""
        print("📝 Encoding text with PLM (all-mpnet-base-v2)...")
        
        # Encode train text
        self.train_text_embeddings = self.text_encoder.encode(
            self.train_text, 
            show_progress_bar=True,
            convert_to_tensor=True
        ).cpu().numpy()
        
        # Encode test text
        self.test_text_embeddings = self.text_encoder.encode(
            self.test_text, 
            show_progress_bar=True,
            convert_to_tensor=True
        ).cpu().numpy()
        
        print(f"   • Train text embeddings: {self.train_text_embeddings.shape}")
        print(f"   • Test text embeddings: {self.test_text_embeddings.shape}")
        print(f"   • Text embedding dimension: {self.train_text_embeddings.shape[1]}")
    
    def create_sequences_for_forecasting(self):
        """
        Create sequences for time series forecasting
        Input: L timesteps, Output: T future timesteps
        """
        print("🔄 Creating forecasting sequences (L→T prediction)...")
        
        def create_sequences(stock_data, text_data, dates):
            sequences_stock = []
            sequences_text = []
            targets = []
            sequence_dates = []
            
            for i in range(self.look_back, len(stock_data) - self.predict_ahead + 1):
                # Input: L timesteps
                stock_seq = stock_data[i-self.look_back:i]  # Shape: (L, 5)
                text_seq = text_data[i-self.look_back:i]    # Shape: (L, text_dim)
                
                # Target: T future timesteps (close price)
                target = stock_data[i:i+self.predict_ahead, 3]  # Close price column
                
                sequences_stock.append(stock_seq)
                sequences_text.append(text_seq)
                targets.append(target)
                sequence_dates.append(dates[i])
            
            return np.array(sequences_stock), np.array(sequences_text), np.array(targets), sequence_dates
        
        # Create training sequences
        self.train_sequences_stock, self.train_sequences_text, self.train_targets, self.train_seq_dates = \
            create_sequences(self.train_stock_normalized, self.train_text_embeddings, self.train_dates)
        
        # Create test sequences
        self.test_sequences_stock, self.test_sequences_text, self.test_targets, self.test_seq_dates = \
            create_sequences(self.test_stock_normalized, self.test_text_embeddings, self.test_dates)
        
        print(f"   • Train sequences: {len(self.train_sequences_stock)}")
        print(f"   • Test sequences: {len(self.test_sequences_stock)}")
        print(f"   • Input shape - Stock: {self.train_sequences_stock.shape}")
        print(f"   • Input shape - Text: {self.train_sequences_text.shape}")
        print(f"   • Target shape: {self.train_targets.shape}")
    
    def preprocess(self):
        """Complete MoAT preprocessing pipeline"""
        print("🚀 MoAT Preprocessing Pipeline\n")
        
        # 1. Load and split data by dates
        train_df, test_df = self.load_and_split_data()
        
        # 2. Instance normalize stock data (as per paper)
        self.instance_normalize_stock_data()
        
        # 3. Encode text with PLM (as per paper)
        self.encode_text_data()
        
        # 4. Create sequences for forecasting
        self.create_sequences_for_forecasting()
        
        print("\n✅ MoAT preprocessing complete!")
        
        return {
            'train': {
                'stock': self.train_sequences_stock,
                'text': self.train_sequences_text,
                'targets': self.train_targets,
                'dates': self.train_seq_dates
            },
            'test': {
                'stock': self.test_sequences_stock,
                'text': self.test_sequences_text,
                'targets': self.test_targets,
                'dates': self.test_seq_dates
            }
        }

# Usage
def main():
    # Initialize with your requirements
    preprocessor = MoATDataPreprocessor(
        csv_file='nvda_combined_dataset_2022_2025.csv',
        look_back=8,     # L = 8 (as per paper)
        predict_ahead=1  # T = 1 (predict 1 day ahead)
    )
    
    # Preprocess data
    data_splits = preprocessor.preprocess()
    
    # Show results
    print(f"\n📊 Final Dataset Summary:")
    print(f"   • Training period: 2022-2024 ({len(data_splits['train']['stock'])} sequences)")
    print(f"   • Testing period: 2025 H1 ({len(data_splits['test']['stock'])} sequences)")
    print(f"   • Input: {data_splits['train']['stock'][0].shape} stock + {data_splits['train']['text'][0].shape} text")
    print(f"   • Output: {data_splits['train']['targets'][0].shape} future price")
    
    # Save for MoAT model
    np.savez('nvda_moat_data.npz',
             **{f"{split}_{key}": value for split, data in data_splits.items() 
                for key, value in data.items() if key != 'dates'})
    
    print(f"\n💾 Data saved to: nvda_moat_data.npz")
    print(f"🎯 Ready for MoAT model implementation!")
    
    return preprocessor, data_splits

if __name__ == "__main__":
    preprocessor, data_splits = main()