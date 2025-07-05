import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import the MoAT model
from moat_architecture import MoATModel, MoATDataset, create_moat_model

class MoATTrainer:
    """MoAT Training Pipeline"""
    
    def __init__(self, model, train_loader, test_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.0001,
            weight_decay=0.001
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=30, 
            gamma=0.7
        )
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = float('inf')
        
        print(f"🚂 MoAT Trainer Ready:")
        print(f"   • Device: {device}")
        print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • Train batches: {len(train_loader)}")
        print(f"   • Test batches: {len(test_loader)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            stock = batch['stock'].to(self.device)
            text = batch['text'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            predictions, individual_preds = self.model(stock, text)
            
            # Main loss
            main_loss = self.criterion(predictions, targets)
            
            # Individual losses for diversity
            individual_losses = [self.criterion(pred.squeeze(), targets) for pred in individual_preds]
            individual_loss = torch.stack(individual_losses).mean()
            
            # Combined loss
            loss = main_loss + 0.1 * individual_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                stock = batch['stock'].to(self.device)
                text = batch['text'].to(self.device)
                targets = batch['target'].to(self.device)
                
                predictions, _ = self.model(stock, text)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def train(self, num_epochs=150, save_path='moat_best_model.pth'):
        """Complete training loop"""
        print(f"\n🚀 Training MoAT for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.scheduler.step()
            
            # Save best model
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_losses': self.train_losses,
                    'test_losses': self.test_losses
                }, save_path)
                best_epoch = epoch
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | Best: {self.best_test_loss:.6f}")
        
        print(f"\n✅ Training Complete!")
        print(f"   • Best test loss: {self.best_test_loss:.6f} at epoch {best_epoch+1}")
        print(f"   • Model saved: {save_path}")
        
        self.plot_training_history()
        return self.train_losses, self.test_losses
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(self.test_losses, label='Test Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MoAT Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Smoothed version
        plt.subplot(1, 2, 2)
        if len(self.train_losses) > 10:
            # Simple moving average
            window = 10
            train_smooth = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            test_smooth = np.convolve(self.test_losses, np.ones(window)/window, mode='valid')
            epochs_smooth = range(window-1, len(self.train_losses))
            
            plt.plot(epochs_smooth, train_smooth, label='Train Loss (Smoothed)', color='blue')
            plt.plot(epochs_smooth, test_smooth, label='Test Loss (Smoothed)', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Smoothed Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('moat_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_data_loaders(batch_size=32):
    """Create data loaders"""
    print("📊 Loading preprocessed data...")
    data = np.load('nvda_moat_data.npz')
    
    train_dataset = MoATDataset(data['train_stock'], data['train_text'], data['train_targets'])
    test_dataset = MoATDataset(data['test_stock'], data['test_text'], data['test_targets'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"   • Train: {len(train_dataset)} samples")
    print(f"   • Test: {len(test_dataset)} samples")
    
    return train_loader, test_loader

def main():
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create data and model
    train_loader, test_loader = create_data_loaders(batch_size=32)
    model = create_moat_model(stock_dim=5, text_dim=768)
    
    # Train
    trainer = MoATTrainer(model, train_loader, test_loader, device)
    trainer.train(num_epochs=150, save_path='moat_best_model.pth')

if __name__ == "__main__":
    main()