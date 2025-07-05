import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import MoAT components
from moat_architecture import MoATModel, MoATDataset, create_moat_model

class MoATEvaluator:
    """Robust MoAT Model Evaluation"""
    
    def __init__(self, model_path='moat_best_model.pth', device='cpu'):
        self.device = device
        self.model_path = model_path
        
        # Load model
        self.model = create_moat_model(stock_dim=5, text_dim=768)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load original data
        self.df = pd.read_csv('nvda_combined_dataset_2022_2025.csv')
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        print(f"📊 MoAT Evaluator Ready:")
        print(f"   • Model: {model_path}")
        print(f"   • Device: {device}")
        print(f"   • Training loss: {checkpoint.get('test_loss', 'N/A'):.6f}")
    
    def predict(self, data_loader, desc="Predicting"):
        """Generate predictions without denormalization complexity"""
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                stock = batch['stock'].to(self.device)
                text = batch['text'].to(self.device)
                target = batch['target'].to(self.device)
                
                pred, _ = self.model(stock, text)
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        return np.array(predictions).flatten(), np.array(targets).flatten()
    
    def denormalize_predictions(self, normalized_preds, original_data, data_type='test'):
        """Denormalize predictions to actual stock prices"""
        # Use close price column (index 3) for denormalization
        close_prices = original_data[:, 3]  # Close price column
        
        # Calculate normalization parameters
        mi = np.mean(close_prices)
        sigma_i = np.std(close_prices)
        
        # Denormalize: actual = normalized * sigma + mi
        if sigma_i > 0:
            denormalized = normalized_preds * sigma_i + mi
        else:
            denormalized = normalized_preds + mi
            
        return denormalized
    
    def calculate_robust_metrics(self, y_true, y_pred, data_name=""):
        """Calculate metrics with robust error handling"""
        
        # Ensure 1D arrays and same length
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        print(f"   📊 Calculating {data_name} metrics for {len(y_true)} samples")
        
        if len(y_true) == 0:
            print(f"   ⚠️  Warning: No data for {data_name}")
            return self._get_empty_metrics()
        
        # Basic regression metrics
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
        except Exception as e:
            print(f"   ❌ Error calculating basic metrics: {e}")
            return self._get_empty_metrics()
        
        # MAPE with robust handling
        try:
            # Avoid division by zero
            non_zero_mask = np.abs(y_true) > 1e-8
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = float('inf')
        except:
            mape = float('inf')
        
        # SMAPE
        try:
            denominator = np.abs(y_pred) + np.abs(y_true)
            smape = np.mean(2 * np.abs(y_pred - y_true) / np.maximum(denominator, 1e-8)) * 100
        except:
            smape = float('inf')
        
        # R-squared
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0
        except:
            r2 = 0
        
        # Directional accuracy with robust handling
        try:
            if len(y_true) > 1:
                y_true_diff = np.diff(y_true)
                y_pred_diff = np.diff(y_pred)
                
                # Ensure both diff arrays exist and have same length
                if len(y_true_diff) > 0 and len(y_pred_diff) > 0:
                    min_diff_len = min(len(y_true_diff), len(y_pred_diff))
                    y_true_diff = y_true_diff[:min_diff_len]
                    y_pred_diff = y_pred_diff[:min_diff_len]
                    
                    # Calculate directional accuracy
                    correct_directions = np.sum(
                        (y_true_diff > 0) == (y_pred_diff > 0)
                    )
                    directional_accuracy = (correct_directions / len(y_true_diff)) * 100
                else:
                    directional_accuracy = 0
            else:
                directional_accuracy = 0
        except Exception as e:
            print(f"   ⚠️  Error calculating directional accuracy: {e}")
            directional_accuracy = 0
        
        # Additional metrics
        try:
            max_error = np.max(np.abs(y_true - y_pred))
            mean_true = np.mean(y_true)
            mean_pred = np.mean(y_pred)
            std_true = np.std(y_true)
            std_pred = np.std(y_pred)
        except:
            max_error = mean_true = mean_pred = std_true = std_pred = 0
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'R²': r2,
            'Directional_Accuracy': directional_accuracy,
            'Max_Error': max_error,
            'Mean_True': mean_true,
            'Mean_Pred': mean_pred,
            'Std_True': std_true,
            'Std_Pred': std_pred,
            'Sample_Count': len(y_true)
        }
    
    def _get_empty_metrics(self):
        """Return empty metrics dict"""
        return {
            'MSE': 0, 'MAE': 0, 'RMSE': 0,
            'MAPE': float('inf'), 'SMAPE': float('inf'), 'R²': 0,
            'Directional_Accuracy': 0, 'Max_Error': 0,
            'Mean_True': 0, 'Mean_Pred': 0,
            'Std_True': 0, 'Std_Pred': 0,
            'Sample_Count': 0
        }
    
    def create_comprehensive_evaluation(self):
        """Create comprehensive evaluation with robust error handling"""
        print("📈 Creating comprehensive evaluation...")
        
        # Load data
        try:
            data = np.load('nvda_moat_data.npz')
            print(f"   📊 Data loaded: Train {data['train_stock'].shape}, Test {data['test_stock'].shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
        
        # Create datasets and loaders
        try:
            train_dataset = MoATDataset(data['train_stock'], data['train_text'], data['train_targets'])
            test_dataset = MoATDataset(data['test_stock'], data['test_text'], data['test_targets'])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
            
            print(f"   📦 Datasets created: Train {len(train_dataset)}, Test {len(test_dataset)}")
        except Exception as e:
            print(f"❌ Error creating datasets: {e}")
            return None, None
        
        # Get predictions
        try:
            print("   🚂 Getting training predictions...")
            train_pred, train_true = self.predict(train_loader, "Train Predictions")
            
            print("   🧪 Getting test predictions...")
            test_pred, test_true = self.predict(test_loader, "Test Predictions")
            
            print(f"   📏 Shapes: Train pred {train_pred.shape}, Test pred {test_pred.shape}")
        except Exception as e:
            print(f"❌ Error getting predictions: {e}")
            return None, None
        
        # Calculate metrics
        try:
            train_metrics = self.calculate_robust_metrics(train_true, train_pred, "Training")
            test_metrics = self.calculate_robust_metrics(test_true, test_pred, "Test")
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return None, None
        
        # Create visualization
        try:
            self.create_enhanced_plots(train_true, train_pred, test_true, test_pred, train_metrics, test_metrics, data)
        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")
        
        return train_metrics, test_metrics
    
    def create_enhanced_plots(self, train_true, train_pred, test_true, test_pred, train_metrics, test_metrics, data):
        """Create enhanced visualization plots"""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # 1. Enhanced NVIDIA Stock Price Comparison with Actual vs Predicted
        ax1 = axes[0, 0]
        
        # Get date ranges for plotting
        train_mask = (self.df['date'] >= '2022-01-01') & (self.df['date'] <= '2024-12-31')
        test_mask = (self.df['date'] >= '2025-01-01') & (self.df['date'] <= '2025-06-30')
        
        train_df = self.df[train_mask].copy()
        test_df = self.df[test_mask].copy()
        
        # Plot training data in blue
        ax1.plot(train_df['date'], train_df['close'], 
                color='blue', alpha=0.8, label='Training Data (2022-2024)', linewidth=2)
        
        # For test period, we need to denormalize predictions to actual stock prices
        # Get original test data for denormalization
        test_stock_data = data['test_stock']  # This is normalized data
        
        # We need to denormalize the predictions
        # First, get the normalization parameters from test data
        # The predictions correspond to sequences, so we need to align them properly
        
        # Get the actual test close prices from the CSV
        actual_test_prices = test_df['close'].values
        
        # The predictions are for sequences starting from look_back positions
        # So we need to align them properly
        look_back = 8  # This should match the look_back used in preprocessing
        
        # Create date range for predictions (they start from look_back positions)
        if len(actual_test_prices) > look_back:
            pred_dates = test_df['date'].iloc[look_back:look_back+len(test_pred)]
            actual_dates = test_df['date'].iloc[look_back:look_back+len(test_pred)]
            actual_prices_aligned = actual_test_prices[look_back:look_back+len(test_pred)]
            
            # For denormalization, we can use the relationship between normalized targets and actual prices
            # Since test_true are normalized close prices and we have actual close prices
            if len(test_true) > 0 and len(actual_prices_aligned) > 0:
                # Calculate denormalization parameters from the test data
                # actual_prices = normalized_prices * scale + offset
                scale = np.std(actual_prices_aligned)
                offset = np.mean(actual_prices_aligned)
                
                # Denormalize predictions
                pred_prices = test_pred * scale + offset
                
                # Plot actual test values in green
                ax1.plot(actual_dates, actual_prices_aligned,
                        color='green', alpha=0.8, label='Actual Test Values (2025)', linewidth=2)
                
                # Plot predicted test values in red
                ax1.plot(pred_dates, pred_prices,
                        color='red', alpha=0.8, label='Predicted Test Values (2025)', linewidth=2, linestyle='--')
            else:
                print("⚠️ Could not align predictions with dates properly")
        
        ax1.set_title('NVIDIA Stock Price: Training vs Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Training predictions (normalized)
        ax2 = axes[0, 1]
        sample_size = min(200, len(train_pred))
        indices = np.linspace(0, len(train_pred)-1, sample_size, dtype=int)
        
        ax2.plot(train_true[indices], color='blue', label='Actual', linewidth=2, alpha=0.8)
        ax2.plot(train_pred[indices], color='red', label='Predicted', linewidth=2, alpha=0.7)
        ax2.set_title(f'Training: Actual vs Predicted (Normalized)\nMAPE: {train_metrics["MAPE"]:.2f}%, R²: {train_metrics["R²"]:.4f}')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Normalized Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Test predictions (normalized)
        ax3 = axes[1, 0]
        ax3.plot(test_true, color='green', label='Actual', linewidth=2, alpha=0.8)
        ax3.plot(test_pred, color='red', label='Predicted', linewidth=2, alpha=0.7)
        ax3.set_title(f'Test: Actual vs Predicted (Normalized)\nMAPE: {test_metrics["MAPE"]:.2f}%, R²: {test_metrics["R²"]:.4f}')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Normalized Price')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter plot
        ax4 = axes[1, 1]
        ax4.scatter(test_true, test_pred, alpha=0.6, color='green', s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(np.min(test_true), np.min(test_pred))
        max_val = max(np.max(test_true), np.max(test_pred))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Actual Values (Normalized)')
        ax4.set_ylabel('Predicted Values (Normalized)')
        ax4.set_title(f'Prediction Accuracy\nR² = {test_metrics["R²"]:.4f}')
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals
        ax5 = axes[2, 0]
        residuals = test_true - test_pred
        ax5.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax5.axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
        ax5.set_xlabel('Residuals (Actual - Predicted)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Residual Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance summary
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Diagnose issues
        issues = []
        if test_metrics['MAPE'] > 100:
            issues.append("🔴 CRITICAL: MAPE > 100%")
        if test_metrics['R²'] < 0:
            issues.append("🔴 CRITICAL: Negative R²")
        if test_metrics['Std_Pred'] < test_metrics['Std_True'] * 0.1:
            issues.append("🔴 CRITICAL: Predicting constants")
        
        performance_text = f"""
MODEL PERFORMANCE ANALYSIS

Training Set:
- MAPE: {train_metrics['MAPE']:.2f}%
- R²: {train_metrics['R²']:.4f}
- MAE: {train_metrics['MAE']:.4f}
- Samples: {train_metrics['Sample_Count']}

Test Set (2025):
- MAPE: {test_metrics['MAPE']:.2f}%
- R²: {test_metrics['R²']:.4f}
- MAE: {test_metrics['MAE']:.4f}
- Direction Acc: {test_metrics['Directional_Accuracy']:.1f}%
- Samples: {test_metrics['Sample_Count']}

Prediction Variability:
- Actual Std: {test_metrics['Std_True']:.4f}
- Predicted Std: {test_metrics['Std_Pred']:.4f}

Issues Detected:
"""
        
        for issue in issues:
            performance_text += f"\n{issue}"
        
        if not issues:
            performance_text += "\n🟢 No critical issues detected"
        
        ax6.text(0.05, 0.95, performance_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('moat_robust_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_comprehensive_results(self, train_metrics, test_metrics):
        """Print detailed results and diagnosis"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE MOAT MODEL EVALUATION")
        print("="*80)
        
        print(f"\n📊 TRAINING PERFORMANCE:")
        print(f"{'Metric':<25} {'Value':<15} {'Status'}")
        print("-" * 65)
        print(f"{'MSE':<25} {train_metrics['MSE']:<15.6f} {'✅' if train_metrics['MSE'] < 1 else '⚠️'}")
        print(f"{'MAE':<25} {train_metrics['MAE']:<15.6f} {'✅' if train_metrics['MAE'] < 0.5 else '⚠️'}")
        print(f"{'MAPE':<25} {train_metrics['MAPE']:<15.2f}% {'✅' if train_metrics['MAPE'] < 20 else '❌'}")
        print(f"{'R²':<25} {train_metrics['R²']:<15.4f} {'✅' if train_metrics['R²'] > 0.5 else '❌'}")
        print(f"{'Direction Accuracy':<25} {train_metrics['Directional_Accuracy']:<15.2f}% {'✅' if train_metrics['Directional_Accuracy'] > 50 else '❌'}")
        
        print(f"\n📈 TEST PERFORMANCE (2025 Predictions):")
        print(f"{'Metric':<25} {'Value':<15} {'Status'}")
        print("-" * 65)
        print(f"{'MSE':<25} {test_metrics['MSE']:<15.6f} {'✅' if test_metrics['MSE'] < 1 else '⚠️'}")
        print(f"{'MAE':<25} {test_metrics['MAE']:<15.6f} {'✅' if test_metrics['MAE'] < 0.5 else '⚠️'}")
        print(f"{'MAPE':<25} {test_metrics['MAPE']:<15.2f}% {'✅' if test_metrics['MAPE'] < 20 else '❌'}")
        print(f"{'R²':<25} {test_metrics['R²']:<15.4f} {'✅' if test_metrics['R²'] > 0.5 else '❌'}")
        print(f"{'Direction Accuracy':<25} {test_metrics['Directional_Accuracy']:<15.2f}% {'✅' if test_metrics['Directional_Accuracy'] > 50 else '❌'}")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        if test_metrics['MAPE'] > 100:
            print("🔴 POOR: Model has very high error rate (>100% MAPE)")
            print("   💡 Recommendations:")
            print("   • Retrain with learning rate 0.00001")
            print("   • Increase epochs to 300+")
            print("   • Check data preprocessing")
        elif test_metrics['MAPE'] > 50:
            print("🟠 FAIR: Model needs improvement")
        elif test_metrics['MAPE'] < 20 and test_metrics['R²'] > 0.6:
            print("🟢 GOOD: Model shows decent performance")
        else:
            print("🟡 MODERATE: Model has mixed performance")
        
        print("="*80)

def main():
    """Main evaluation with robust error handling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        evaluator = MoATEvaluator('moat_best_model.pth', device)
        train_metrics, test_metrics = evaluator.create_comprehensive_evaluation()
        
        if train_metrics and test_metrics:
            evaluator.print_comprehensive_results(train_metrics, test_metrics)
        else:
            print("❌ Evaluation failed - check error messages above")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("🔧 Check that all required files exist:")
        print("   • moat_best_model.pth")
        print("   • nvda_moat_data.npz") 
        print("   • nvda_combined_dataset_2022_2025.csv")

if __name__ == "__main__":
    main()