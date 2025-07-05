import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

class MoATDataset(Dataset):
    """Dataset class for MoAT model"""
    def __init__(self, stock_sequences, text_sequences, targets):
        self.stock_sequences = torch.FloatTensor(stock_sequences)
        self.text_sequences = torch.FloatTensor(text_sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.stock_sequences)
    
    def __getitem__(self, idx):
        return {
            'stock': self.stock_sequences[idx],
            'text': self.text_sequences[idx],
            'target': self.targets[idx]
        }

class PatchEmbedding(nn.Module):
    """Patch-wise embedding for time series"""
    def __init__(self, input_dim, d_model, patch_len=4, stride=2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.projection = nn.Linear(patch_len * input_dim, d_model)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Returns: (batch_size, num_patches, d_model)
        """
        batch_size, seq_len, input_dim = x.shape
        
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :].reshape(batch_size, -1)
            patches.append(patch)
        
        if patches:
            patches = torch.stack(patches, dim=1)
            embeddings = self.projection(patches)
        else:
            embeddings = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        
        return embeddings

class TextPatchEmbedding(nn.Module):
    """Fixed Text patch embedding to match time series patches"""
    def __init__(self, text_dim, d_model, patch_len=4, stride=2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.text_dim = text_dim
        self.d_model = d_model
        
        # Attention parameters for attentive pooling (as per paper Eq. 1)
        self.W = nn.Linear(text_dim, d_model)
        self.b = nn.Parameter(torch.zeros(d_model))
        self.V = nn.Linear(d_model, 1)  # Output scalar attention weights
        
        # Project text to d_model space
        self.text_projection = nn.Linear(text_dim, d_model)
        
    def forward(self, text_embeddings):
        """
        text_embeddings: (batch_size, seq_len, text_dim)
        Returns: (batch_size, num_patches, d_model) - same as time series patches
        """
        batch_size, seq_len, text_dim = text_embeddings.shape
        
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            # Get text patch: (batch, patch_len, text_dim)
            text_patch = text_embeddings[:, i:i+self.patch_len, :]
            
            # Apply attentive pooling within the patch
            # Following paper Eq. 1: Softmax(tanh(d * W + b) * V) * d
            
            # Reshape for batch processing: (batch * patch_len, text_dim)
            text_flat = text_patch.reshape(batch_size * self.patch_len, text_dim)
            
            # Compute attention scores
            attention_input = torch.tanh(self.W(text_flat) + self.b)  # (batch*patch_len, d_model)
            attention_scores = self.V(attention_input)  # (batch*patch_len, 1)
            attention_scores = attention_scores.reshape(batch_size, self.patch_len, 1)  # (batch, patch_len, 1)
            
            # Apply softmax over patch dimension
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, patch_len, 1)
            
            # Apply attention to original text embeddings
            attended_text = (attention_weights * text_patch).sum(dim=1)  # (batch, text_dim)
            
            # Project to d_model space
            patch_embedded = self.text_projection(attended_text)  # (batch, d_model)
            patches.append(patch_embedded)
        
        if patches:
            patches = torch.stack(patches, dim=1)  # (batch, num_patches, d_model)
        else:
            patches = torch.zeros(batch_size, 1, self.d_model, device=text_embeddings.device)
        
        return patches

class TrendSeasonalDecomposition(nn.Module):
    """Trend-Seasonal decomposition"""
    def __init__(self, moving_avg_kernel=3):
        super().__init__()
        self.moving_avg_kernel = moving_avg_kernel
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, features)
        Returns: trend, seasonal components
        """
        batch_size, seq_len, features = x.shape
        
        # Simple moving average for trend
        padding = self.moving_avg_kernel // 2
        x_padded = F.pad(x, (0, 0, padding, padding), mode='reflect')
        
        trends = []
        for i in range(features):
            feature_data = x_padded[:, :, i].unsqueeze(1)  # (batch, 1, padded_seq)
            kernel = torch.ones(1, 1, self.moving_avg_kernel, device=x.device) / self.moving_avg_kernel
            trend = F.conv1d(feature_data, kernel, padding=0)
            trends.append(trend.squeeze(1))
        
        trend = torch.stack(trends, dim=-1)
        seasonal = x - trend
        
        return trend, seasonal

class MultiModalAugmentedEncoder(nn.Module):
    """Multi-modal augmented encoder"""
    def __init__(self, d_model, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.d_model = d_model
        
        # Positional embeddings
        self.pos_embedding_time = nn.Parameter(torch.randn(100, d_model))
        self.pos_embedding_text = nn.Parameter(torch.randn(100, d_model))
        
        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, time_patches, text_patches):
        """
        Both inputs should have same shape: (batch_size, num_patches, d_model)
        """
        batch_size, num_patches, d_model = time_patches.shape
        
        # Add positional embeddings
        time_with_pos = time_patches + self.pos_embedding_time[:num_patches].unsqueeze(0)
        text_with_pos = text_patches + self.pos_embedding_text[:num_patches].unsqueeze(0)
        
        # Sample-wise augmentation (separate processing)
        time_encoded = self.transformer_encoder(time_with_pos)
        text_encoded = self.transformer_encoder(text_with_pos)
        
        # Feature-wise augmentation (joint processing)
        joint_input = torch.cat([time_with_pos, text_with_pos], dim=1)
        joint_encoded = self.transformer_encoder(joint_input)
        
        # Split joint encoding back
        time_cross, text_cross = torch.chunk(joint_encoded, 2, dim=1)
        
        return {
            'time_single': time_encoded,
            'text_single': text_encoded,
            'time_cross': time_cross,
            'text_cross': text_cross
        }

class CrossModalFusion(nn.Module):
    """Cross-modal fusion with multiple decoders"""
    def __init__(self, d_model, output_dim=1):
        super().__init__()
        self.trend_decoder = nn.Linear(d_model, output_dim)
        self.seasonal_decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, trend_representations, seasonal_representations):
        """Generate all 16 combinations"""
        predictions = []
        
        for z_T in trend_representations:
            for z_S in seasonal_representations:
                # Global average pooling
                trend_pooled = z_T.mean(dim=1)    # (batch, d_model)
                seasonal_pooled = z_S.mean(dim=1) # (batch, d_model)
                
                # Decode
                trend_pred = self.trend_decoder(trend_pooled)
                seasonal_pred = self.seasonal_decoder(seasonal_pooled)
                
                # Combine
                combined_pred = trend_pred + seasonal_pred
                predictions.append(combined_pred)
        
        return predictions

class MoATModel(nn.Module):
    """Complete MoAT model implementation"""
    def __init__(self, stock_dim=5, text_dim=768, d_model=64, patch_len=4, stride=2):
        super().__init__()
        
        # Patch embeddings
        self.stock_patch_embedding = PatchEmbedding(stock_dim, d_model, patch_len, stride)
        self.text_patch_embedding = TextPatchEmbedding(text_dim, d_model, patch_len, stride)
        
        # Trend-seasonal decomposition
        self.decomposition = TrendSeasonalDecomposition()
        
        # Multi-modal encoder
        self.encoder = MultiModalAugmentedEncoder(d_model)
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(d_model)
        
        # Prediction synthesis
        self.synthesis_weights = nn.Parameter(torch.ones(16) / 16)
        self.synthesis_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, stock_sequences, text_sequences):
        """
        stock_sequences: (batch_size, L, stock_dim)
        text_sequences: (batch_size, L, text_dim)
        """
        
        # 1. Trend-seasonal decomposition
        stock_trend, stock_seasonal = self.decomposition(stock_sequences)
        
        # 2. Patch embeddings - now both will have same shape
        stock_trend_patches = self.stock_patch_embedding(stock_trend)
        stock_seasonal_patches = self.stock_patch_embedding(stock_seasonal)
        text_trend_patches = self.text_patch_embedding(text_sequences)
        text_seasonal_patches = self.text_patch_embedding(text_sequences)
        
        print(f"🔍 Debug - After patch embedding:")
        print(f"   • stock_trend_patches: {stock_trend_patches.shape}")
        print(f"   • text_trend_patches: {text_trend_patches.shape}")
        
        # 3. Multi-modal encoding
        trend_encoded = self.encoder(stock_trend_patches, text_trend_patches)
        seasonal_encoded = self.encoder(stock_seasonal_patches, text_seasonal_patches)
        
        # 4. Prepare representations for fusion
        trend_reps = [
            trend_encoded['time_single'], trend_encoded['text_single'],
            trend_encoded['time_cross'], trend_encoded['text_cross']
        ]
        seasonal_reps = [
            seasonal_encoded['time_single'], seasonal_encoded['text_single'],
            seasonal_encoded['time_cross'], seasonal_encoded['text_cross']
        ]
        
        # 5. Cross-modal fusion (16 predictions)
        predictions = self.fusion(trend_reps, seasonal_reps)
        
        # 6. Prediction synthesis
        stacked_predictions = torch.stack(predictions, dim=-1)  # (batch, 1, 16)
        final_prediction = torch.sum(stacked_predictions * self.synthesis_weights, dim=-1) + self.synthesis_bias
        
        return final_prediction.squeeze(-1), predictions

def create_moat_model(stock_dim=5, text_dim=768):
    """Create MoAT model instance"""
    model = MoATModel(
        stock_dim=stock_dim,
        text_dim=text_dim,
        d_model=64,
        patch_len=4,
        stride=2
    )
    return model

def test_moat_model():
    """Test MoAT model with sample data"""
    # Load preprocessed data
    data = np.load('nvda_moat_data.npz')
    
    # Create sample batch
    batch_size = 4
    sample_stock = torch.FloatTensor(data['train_stock'][:batch_size])
    sample_text = torch.FloatTensor(data['train_text'][:batch_size])
    
    print(f"📊 Testing Fixed MoAT model:")
    print(f"   • Input stock shape: {sample_stock.shape}")
    print(f"   • Input text shape: {sample_text.shape}")
    
    # Create model
    model = create_moat_model(stock_dim=5, text_dim=768)
    
    # Forward pass
    final_pred, all_predictions = model(sample_stock, sample_text)
    
    print(f"\n✅ MoAT model test successful!")
    print(f"   • Final prediction shape: {final_pred.shape}")
    print(f"   • Number of individual predictions: {len(all_predictions)}")
    print(f"   • Individual prediction shape: {all_predictions[0].shape}")
    print(f"   • Sample predictions: {final_pred[:3].detach().numpy()}")
    
    return model

if __name__ == "__main__":
    model = test_moat_model()