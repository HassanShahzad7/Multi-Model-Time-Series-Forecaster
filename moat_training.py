import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from moat_model import MoATDataset, create_moat_model, PredictionSynthesis

class MoATTrainer:
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader  # Use test as validation
        self.device = device
        
        # Optimizers with better hyperparameters
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.synthesis = PredictionSynthesis(16).to(device)
        self.synthesis_optimizer = optim.Adam(self.synthesis.parameters(), lr=0.01)
        
        self.best_test_loss = float('inf')
        self.train_losses = []
        self.test_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            time_series = batch['time_series'].to(self.device)
            text = batch['text'].to(self.device)
            trend = batch['trend'].to(self.device)
            seasonal = batch['seasonal'].to(self.device)
            targets = batch['target'].to(self.device)
            
            predictions = self.model(time_series, text, trend, seasonal)
            loss = self.model.compute_loss(predictions, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                time_series = batch['time_series'].to(self.device)
                text = batch['text'].to(self.device)
                trend = batch['trend'].to(self.device)
                seasonal = batch['seasonal'].to(self.device)
                targets = batch['target'].to(self.device)
                
                predictions = self.model(time_series, text, trend, seasonal)
                loss = self.model.compute_loss(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Train synthesis on test predictions
        self.train_synthesis(torch.cat(all_predictions), torch.cat(all_targets))
        
        return total_loss / len(self.test_loader)
    
    def train_synthesis(self, predictions, targets, epochs=50):
        self.synthesis.train()
        for _ in range(epochs):
            synthesized = self.synthesis(predictions)
            loss = F.mse_loss(synthesized.squeeze(), targets)
            self.synthesis_optimizer.zero_grad()
            loss.backward()
            self.synthesis_optimizer.step()
    
    def train(self, num_epochs=100):
        print(f"\n🚀 Training MoAT for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'synthesis_state_dict': self.synthesis.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }, 'moat_model.pth')
            
            if (epoch + 1) % 5 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train: {train_loss:.6f} | "
                      f"Test: {test_loss:.6f} | "
                      f"Best: {self.best_test_loss:.6f} | "
                      f"LR: {lr:.6f}")
        
        print(f"\n✅ Training complete! Best test loss: {self.best_test_loss:.6f}")

def main():
    data = np.load('moat_preprocessed_data.npz', allow_pickle=True)
    
    # Only train and test datasets
    train_dataset = MoATDataset(data['train'].item(), include_dates=False)
    test_dataset = MoATDataset(data['test'].item(), include_dates=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model = create_moat_model(
        feature_dim=data['feature_dim'].item(),
        num_patches=data['num_patches'].item(),
        patch_len=data['patch_len'].item()
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = MoATTrainer(model, train_loader, test_loader, device)
    trainer.train(num_epochs=100)

if __name__ == "__main__":
    main()