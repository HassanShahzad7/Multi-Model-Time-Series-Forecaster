import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MoATDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, include_dates=False):
        self.time_series = torch.FloatTensor(data_dict['time_series'])
        self.text = torch.FloatTensor(data_dict['text'])
        self.trend = torch.FloatTensor(data_dict['trend'])
        self.seasonal = torch.FloatTensor(data_dict['seasonal'])
        self.targets = torch.FloatTensor(data_dict['targets'])
        self.dates = data_dict['dates']
        self.include_dates = include_dates
    
    def __len__(self):
        return len(self.time_series)
    
    def __getitem__(self, idx):
        sample = {
            'time_series': self.time_series[idx],
            'text': self.text[idx],
            'trend': self.trend[idx],
            'seasonal': self.seasonal[idx],
            'target': self.targets[idx]
        }
        if self.include_dates:
            sample['date'] = str(self.dates[idx])
        return sample

class MoATModel(nn.Module):
    def __init__(self, time_dim, text_dim=768, d_model=64, num_patches=3, patch_len=4):
        super().__init__()
        
        # FIXED: Correct input dimensions
        time_input_dim = patch_len * time_dim
        
        # Projections
        self.time_projection = nn.Linear(time_input_dim, d_model)
        self.text_projection = nn.Linear(text_dim, d_model)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        
        # FIXED: Correct decoder input dimension
        decoder_input_dim = num_patches * d_model
        self.trend_decoder = nn.Linear(decoder_input_dim, 1)
        self.seasonal_decoder = nn.Linear(decoder_input_dim, 1)
        
        self.num_patches = num_patches
        self.d_model = d_model
    
    def encode_patches(self, time_patches, text_patches):
        batch_size, num_patches, patch_len, time_dim = time_patches.shape
        
        # Project time patches
        time_flat = time_patches.reshape(batch_size * num_patches, -1)
        time_emb = self.time_projection(time_flat)
        time_emb = time_emb.reshape(batch_size, num_patches, self.d_model)
        
        # Project text patches
        text_emb = self.text_projection(text_patches)
        
        outputs = {}
        
        # Sample-wise (separate)
        time_encoded = self.transformer(time_emb)
        text_encoded = self.transformer(text_emb)
        outputs['time_single'] = time_encoded
        outputs['text_single'] = text_encoded
        
        # Feature-wise (combined)
        combined = torch.cat([time_emb, text_emb], dim=1)
        combined_encoded = self.transformer(combined)
        outputs['time_cross'] = combined_encoded[:, :num_patches]
        outputs['text_cross'] = combined_encoded[:, num_patches:]
        
        return outputs
    
    def forward(self, time_series, text, trend, seasonal):
        batch_size = time_series.size(0)
        
        # Encode trend and seasonal
        trend_outputs = self.encode_patches(trend, text)
        seasonal_outputs = self.encode_patches(seasonal, text)
        
        # Collect representations with correct reshaping
        trend_reps = []
        seasonal_reps = []
        
        for key in ['time_single', 'text_single', 'time_cross', 'text_cross']:
            # FIXED: Ensure correct reshaping
            trend_rep = trend_outputs[key].reshape(batch_size, -1)
            seasonal_rep = seasonal_outputs[key].reshape(batch_size, -1)
            trend_reps.append(trend_rep)
            seasonal_reps.append(seasonal_rep)
        
        # Cross-fusion: 16 predictions
        predictions = []
        for t_rep in trend_reps:
            for s_rep in seasonal_reps:
                trend_pred = self.trend_decoder(t_rep)
                seasonal_pred = self.seasonal_decoder(s_rep)
                pred = trend_pred + seasonal_pred
                predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
    
    def compute_loss(self, predictions, targets):
        targets_expanded = targets.unsqueeze(1).unsqueeze(2).expand_as(predictions)
        return F.mse_loss(predictions, targets_expanded)

class PredictionSynthesis(nn.Module):
    def __init__(self, num_predictions=16):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_predictions) / num_predictions)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, predictions):
        weighted = predictions * self.weights.unsqueeze(0).unsqueeze(2)
        return weighted.sum(dim=1) + self.bias

def create_moat_model(feature_dim, num_patches, patch_len):
    return MoATModel(
        time_dim=feature_dim,
        text_dim=768,
        d_model=64,
        num_patches=num_patches,
        patch_len=patch_len
    )