import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from moat_model import MoATDataset, create_moat_model, PredictionSynthesis
import pickle

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics with proper handling"""
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    # MAPE with better handling for small values
    # Use a threshold to avoid division by very small numbers
    threshold = 0.01  # 1% threshold
    mask = np.abs(targets) > threshold
    
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        # If all values are below threshold, use symmetric MAPE
        smape = np.mean(2 * np.abs(predictions - targets) / 
                       (np.abs(predictions) + np.abs(targets) + 1e-8)) * 100
        mape = smape
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def evaluate_moat():
    # Load preprocessed data
    data = np.load('moat_preprocessed_data.npz', allow_pickle=True)
    test_dataset = MoATDataset(data['test'].item(), include_dates=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load original data for price reconstruction
    df = pd.read_csv('nvda_combined_dataset_2022_2025.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create model
    model = create_moat_model(
        feature_dim=data['feature_dim'].item(),
        num_patches=data['num_patches'].item(),
        patch_len=data['patch_len'].item()
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('moat_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    synthesis = PredictionSynthesis(16).to(device)
    synthesis.load_state_dict(checkpoint['synthesis_state_dict'])
    synthesis.eval()
    
    # Collect predictions
    all_predictions = []
    all_targets = []
    all_dates = []
    
    with torch.no_grad():
        for batch in test_loader:
            predictions = model(
                batch['time_series'].to(device),
                batch['text'].to(device),
                batch['trend'].to(device),
                batch['seasonal'].to(device)
            )
            
            final_pred = synthesis(predictions)
            
            all_predictions.extend(final_pred.squeeze().cpu().numpy())
            all_targets.extend(batch['target'].numpy())
            all_dates.extend(batch['date'])
    
    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)
    
    # Calculate metrics on returns
    return_metrics = calculate_metrics(predictions_np, targets_np)
    
    # Convert dates to datetime
    all_dates = [pd.to_datetime(date) for date in all_dates]
    
    # Get actual prices for these dates
    actual_prices = []
    previous_prices = []
    
    for date in all_dates:
        # Find the row with this date
        mask = df['date'] == date
        if mask.sum() > 0:
            actual_price = df.loc[mask, 'close'].values[0]
            actual_prices.append(actual_price)
            
            # Get previous price
            idx = df[mask].index[0]
            if idx > 0:
                prev_price = df.loc[idx-1, 'close']
                previous_prices.append(prev_price)
            else:
                previous_prices.append(actual_price)
        else:
            actual_prices.append(np.nan)
            previous_prices.append(np.nan)
    
    actual_prices = np.array(actual_prices)
    previous_prices = np.array(previous_prices)
    
    # Calculate predicted prices from predicted returns
    predicted_prices = previous_prices * (1 + predictions_np)
    
    # Calculate price prediction metrics
    valid_mask = ~np.isnan(actual_prices)
    if valid_mask.sum() > 0:
        price_metrics = calculate_metrics(
            predicted_prices[valid_mask], 
            actual_prices[valid_mask]
        )
    else:
        price_metrics = {k: np.nan for k in return_metrics.keys()}
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame({
        'date': all_dates,
        'previous_close': previous_prices,
        'actual_close': actual_prices,
        'predicted_close': predicted_prices,
        'actual_return': targets_np,
        'predicted_return': predictions_np,
        'return_error': targets_np - predictions_np,
        'price_error': actual_prices - predicted_prices,
        'price_error_pct': ((actual_prices - predicted_prices) / actual_prices * 100)
    })
    
    # Save detailed results
    results_df.to_csv('moat_predictions_detailed.csv', index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE', 'R²'],
        'Returns': [return_metrics['MAE'], return_metrics['MSE'], 
                   return_metrics['RMSE'], return_metrics['MAPE'], 
                   return_metrics['R2']],
        'Prices': [price_metrics['MAE'], price_metrics['MSE'], 
                  price_metrics['RMSE'], price_metrics['MAPE'], 
                  price_metrics['R2']]
    })
    metrics_df.to_csv('moat_metrics_comprehensive.csv', index=False)
    
    print("\n📊 MoAT Evaluation Results:")
    print("\n🔄 Return Prediction Metrics:")
    print(f"   • MAE: {return_metrics['MAE']:.6f}")
    print(f"   • MSE: {return_metrics['MSE']:.6f}")
    print(f"   • RMSE: {return_metrics['RMSE']:.6f}")
    print(f"   • MAPE: {return_metrics['MAPE']:.2f}%")
    print(f"   • R²: {return_metrics['R2']:.4f}")
    
    print("\n💰 Price Prediction Metrics:")
    print(f"   • MAE: ${price_metrics['MAE']:.2f}")
    print(f"   • MSE: {price_metrics['MSE']:.2f}")
    print(f"   • RMSE: ${price_metrics['RMSE']:.2f}")
    print(f"   • MAPE: {price_metrics['MAPE']:.2f}%")
    print(f"   • R²: {price_metrics['R2']:.4f}")
    
    # Additional analysis
    print("\n📈 Additional Analysis:")
    print(f"   • Average actual return: {np.mean(targets_np)*100:.2f}%")
    print(f"   • Average predicted return: {np.mean(predictions_np)*100:.2f}%")
    print(f"   • Return prediction std: {np.std(predictions_np):.4f}")
    print(f"   • Directional accuracy: {np.mean(np.sign(predictions_np) == np.sign(targets_np))*100:.1f}%")
    
    print(f"\n💾 Results saved to:")
    print(f"   • moat_predictions_detailed.csv (with actual prices)")
    print(f"   • moat_metrics_comprehensive.csv")
    
    # Plot if needed
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price predictions
        axes[0, 0].plot(results_df['date'], results_df['actual_close'], 
                       label='Actual Price', color='blue', linewidth=2)
        axes[0, 0].plot(results_df['date'], results_df['predicted_close'], 
                       label='Predicted Price', color='red', alpha=0.7, linewidth=2)
        axes[0, 0].set_title('NVDA Stock Price Predictions')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Return predictions scatter
        axes[0, 1].scatter(results_df['actual_return'], results_df['predicted_return'], 
                          alpha=0.6, s=50)
        min_ret = min(results_df['actual_return'].min(), results_df['predicted_return'].min())
        max_ret = max(results_df['actual_return'].max(), results_df['predicted_return'].max())
        axes[0, 1].plot([min_ret, max_ret], [min_ret, max_ret], 'r--', linewidth=2)
        axes[0, 1].set_xlabel('Actual Returns')
        axes[0, 1].set_ylabel('Predicted Returns')
        axes[0, 1].set_title(f'Return Predictions (R² = {return_metrics["R2"]:.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price error distribution
        axes[1, 0].hist(results_df['price_error_pct'].dropna(), bins=30, 
                       alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Price Prediction Error (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Price Prediction Error Distribution')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        axes[1, 1].plot(results_df['date'], 
                       (1 + results_df['actual_return']).cumprod() - 1,
                       label='Actual Cumulative Return', color='blue', linewidth=2)
        axes[1, 1].plot(results_df['date'], 
                       (1 + results_df['predicted_return']).cumprod() - 1,
                       label='Predicted Cumulative Return', color='red', linewidth=2)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].set_title('Cumulative Returns Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('moat_evaluation_plots.png', dpi=150, bbox_inches='tight')
        print(f"   • moat_evaluation_plots.png")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate plots: {e}")
    
    return return_metrics, price_metrics, results_df

if __name__ == "__main__":
    return_metrics, price_metrics, results = evaluate_moat()