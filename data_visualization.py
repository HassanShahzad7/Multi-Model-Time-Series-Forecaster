import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')
    print("✓ Created 'output' directory")

def load_and_process_data(data_dir='data/'):
    """
    Load all CSV files and consolidate them into a single DataFrame
    """
    
    # Define file mappings with their column names
    file_mappings = {
        'actual': {
            'file': 'nvda_combined_dataset_2022_2025.csv',
            'date_col': 'date',
            'close_col': 'close',
            'pred_col': None
        },
        'lr_with_sentiment': {
            'file': 'lr_predictions_with_sentiment.csv',
            'date_col': 'Date',
            'close_col': 'actual_close',
            'pred_col': 'predicted_close'
        },
        'lr_without_sentiment': {
            'file': 'lr_predictions_without_sentiment.csv',
            'date_col': 'Date',
            'close_col': 'actual_close',
            'pred_col': 'predicted_close'
        },
        'lstm_with_sentiment': {
            'file': 'lstm_predictions_with_sentiment.csv',
            'date_col': 'Date',
            'close_col': 'actual_close',
            'pred_col': 'predicted_close'
        },
        'lstm_without_sentiment': {
            'file': 'lstm_predictions_without_sentiment.csv',
            'date_col': 'Date',
            'close_col': 'actual_close',
            'pred_col': 'predicted_close'
        },
        'sarima_with_sentiment': {
            'file': 'sarima_predictions_with_sentiment.csv',
            'date_col': 'Date',
            'close_col': 'actual_close',
            'pred_col': 'predicted_close'
        },
        'sarima_without_sentiment': {
            'file': 'sarima_predictions_without_sentiment.csv',
            'date_col': 'Date',
            'close_col': 'actual_close',
            'pred_col': 'predicted_close'
        },
        'moat': {
            'file': 'moat_predictions_detailed.csv',
            'date_col': 'date',
            'close_col': 'actual_close',
            'pred_col': 'predicted_close'
        }
    }
    
    # Initialize the consolidated dataframe
    consolidated_df = None
    
    # Load each file and merge
    for model_name, config in file_mappings.items():
        try:
            # Read CSV file
            df = pd.read_csv(data_dir + config['file'])
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df[config['date_col']])
            
            # Select relevant columns
            if model_name == 'actual':
                # For actual data, we only need date and close price
                temp_df = df[['date', config['close_col']]].copy()
                temp_df.rename(columns={config['close_col']: 'actual_close'}, inplace=True)
            else:
                # For prediction models, we need date and predicted close
                if config['pred_col'] in df.columns:
                    temp_df = df[['date', config['pred_col']]].copy()
                    temp_df.rename(columns={config['pred_col']: f'pred_{model_name}'}, inplace=True)
                else:
                    print(f"Warning: {config['pred_col']} not found in {config['file']}")
                    continue
            
            # Merge with consolidated dataframe
            if consolidated_df is None:
                consolidated_df = temp_df
            else:
                consolidated_df = pd.merge(consolidated_df, temp_df, on='date', how='outer')
            
            print(f"Loaded {model_name}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error loading {config['file']}: {str(e)}")
    
    # Sort by date
    consolidated_df.sort_values('date', inplace=True)
    
    # Filter for the specified period (June 2023 to December 2024)
    start_date = pd.to_datetime('2023-06-01')
    end_date = pd.to_datetime('2024-12-31')
    consolidated_df = consolidated_df[(consolidated_df['date'] >= start_date) & 
                                      (consolidated_df['date'] <= end_date)]
    
    # Remove predictions before July 11, 2024
    prediction_start = pd.to_datetime('2024-07-11')
    pred_columns = [col for col in consolidated_df.columns if col.startswith('pred_')]
    
    # Set predictions to NaN for dates before July 11, 2024
    mask = consolidated_df['date'] < prediction_start
    consolidated_df.loc[mask, pred_columns] = np.nan
    
    # Reset index
    consolidated_df.reset_index(drop=True, inplace=True)
    
    return consolidated_df

def calculate_metrics(df, model_col, actual_col='actual_close'):
    """
    Calculate performance metrics for a model
    """
    # Filter rows where both actual and predicted values exist
    mask = df[model_col].notna() & df[actual_col].notna()
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    actual = filtered_df[actual_col].values
    predicted = filtered_df[model_col].values
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mae = np.mean(np.abs(predicted - actual))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def create_main_visualization(df):
    """
    Create main visualization comparing all models
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Define highly distinguishable colors for each model
    model_colors = {
        'pred_lr_with_sentiment': '#FF0000',      # Bright Red
        'pred_lr_without_sentiment': '#FF8C00',   # Dark Orange
        'pred_lstm_with_sentiment': '#0000FF',    # Blue
        'pred_lstm_without_sentiment': '#00CED1', # Dark Turquoise
        'pred_sarima_with_sentiment': '#FFD700',  # Gold
        'pred_sarima_without_sentiment': '#32CD32', # Lime Green
        'pred_moat': '#9370DB'                    # Medium Purple
    }
    
    model_labels = {
        'pred_lr_with_sentiment': 'LR with Sentiment',
        'pred_lr_without_sentiment': 'LR without Sentiment',
        'pred_lstm_with_sentiment': 'LSTM with Sentiment',
        'pred_lstm_without_sentiment': 'LSTM without Sentiment',
        'pred_sarima_with_sentiment': 'SARIMA with Sentiment',
        'pred_sarima_without_sentiment': 'SARIMA without Sentiment',
        'pred_moat': 'MOAT Model'
    }
    
    # Plot 1: Full period
    ax1.plot(df['date'], df['actual_close'], 'k-', linewidth=3, label='Actual Price', alpha=0.9)
    
    for col in df.columns:
        if col.startswith('pred_'):
            ax1.plot(df['date'], df[col], '-', 
                    color=model_colors.get(col, 'gray'),
                    linewidth=2, 
                    label=model_labels.get(col, col),
                    alpha=0.8)
    
    # Add vertical line for prediction start
    prediction_start = pd.to_datetime('2024-07-11')
    ax1.axvline(x=prediction_start, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Prediction Start')
    
    ax1.set_title('NVIDIA Stock Price: Actual vs Predicted (All Models)', fontsize=16, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction period only
    pred_df = df[df['date'] >= prediction_start].copy()
    
    ax2.plot(pred_df['date'], pred_df['actual_close'], 'k-', linewidth=3, label='Actual Price', alpha=0.9)
    
    for col in pred_df.columns:
        if col.startswith('pred_'):
            ax2.plot(pred_df['date'], pred_df[col], '-', 
                    color=model_colors.get(col, 'gray'),
                    linewidth=2, 
                    label=model_labels.get(col, col),
                    alpha=0.8)
    
    ax2.set_title('Prediction Period Only (July 11 - Dec 31, 2024)', fontsize=16, pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Stock Price ($)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/stock_predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/stock_predictions_comparison.png")
    plt.close()

def create_error_analysis(df):
    """
    Create error analysis visualization
    """
    # Calculate errors for prediction period
    prediction_start = pd.to_datetime('2024-07-11')
    pred_df = df[df['date'] >= prediction_start].copy()
    pred_columns = [col for col in df.columns if col.startswith('pred_')]
    
    # Prepare error data with consistent ordering
    errors_dict = {}
    model_order = ['lr_with_sentiment', 'lr_without_sentiment', 'lstm_with_sentiment', 
                   'lstm_without_sentiment', 'sarima_with_sentiment', 'sarima_without_sentiment', 'moat']
    
    for col in pred_columns:
        mask = pred_df[col].notna() & pred_df['actual_close'].notna()
        if mask.sum() > 0:
            errors = pred_df.loc[mask, col].values - pred_df.loc[mask, 'actual_close'].values
            model_key = col.replace('pred_', '')
            model_name = model_key.replace('_', ' ').title()
            errors_dict[model_key] = {'name': model_name, 'errors': errors}
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Box plot of errors
    error_data = []
    labels = []
    box_colors = []
    
    color_map = {
        'lr_with_sentiment': '#FF0000',
        'lr_without_sentiment': '#FF8C00',
        'lstm_with_sentiment': '#0000FF',
        'lstm_without_sentiment': '#00CED1',
        'sarima_with_sentiment': '#FFD700',
        'sarima_without_sentiment': '#32CD32',
        'moat': '#9370DB'
    }
    
    for key in model_order:
        if key in errors_dict:
            error_data.append(errors_dict[key]['errors'])
            labels.append(errors_dict[key]['name'])
            box_colors.append(color_map[key])
    
    bp = ax1.boxplot(error_data, labels=labels, patch_artist=True)
    
    # Color each box
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Error Distribution by Model', fontsize=14)
    ax1.set_ylabel('Error ($)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of errors
    for key in model_order:
        if key in errors_dict:
            ax2.hist(errors_dict[key]['errors'], bins=30, alpha=0.6, 
                    label=errors_dict[key]['name'], color=color_map[key])
    ax2.set_title('Error Distribution Histogram', fontsize=14)
    ax2.set_xlabel('Error ($)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics comparison
    metrics_data = []
    model_names = []
    
    for col in pred_columns:
        metrics = calculate_metrics(pred_df, col)
        if not np.isnan(metrics['RMSE']):
            model_name = col.replace('pred_', '').replace('_', ' ').title()
            model_names.append(model_name)
            metrics_data.append([metrics['RMSE'], metrics['MAE'], metrics['MAPE']])
    
    metrics_data = np.array(metrics_data)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax3.bar(x - width, metrics_data[:, 0], width, label='RMSE', color='#FF6B6B')
    ax3.bar(x, metrics_data[:, 1], width, label='MAE', color='#4ECDC4')
    ax3.bar(x + width, metrics_data[:, 2], width, label='MAPE (%)', color='#45B7D1')
    
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Performance Metrics Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series of absolute errors
    # Use the same distinct colors
    error_colors = {
        'lr_with_sentiment': '#FF0000',      # Bright Red
        'lr_without_sentiment': '#FF8C00',   # Dark Orange
        'lstm_with_sentiment': '#0000FF',    # Blue
        'lstm_without_sentiment': '#00CED1', # Dark Turquoise
        'sarima_with_sentiment': '#FFD700',  # Gold
        'sarima_without_sentiment': '#32CD32', # Lime Green
        'moat': '#9370DB'                    # Medium Purple
    }
    
    for col in pred_columns:
        mask = pred_df[col].notna() & pred_df['actual_close'].notna()
        if mask.sum() > 0:
            abs_errors = np.abs(pred_df.loc[mask, col].values - pred_df.loc[mask, 'actual_close'].values)
            model_name = col.replace('pred_', '').replace('_', ' ').title()
            model_key = col.replace('pred_', '')
            color = error_colors.get(model_key, 'gray')
            ax4.plot(pred_df.loc[mask, 'date'], abs_errors, '-', 
                    label=model_name, color=color, linewidth=2, alpha=0.8)
    
    ax4.set_title('Absolute Errors Over Time', fontsize=14)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Absolute Error ($)', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/error_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/error_analysis.png")
    plt.close()

def create_performance_summary(df):
    """
    Create a performance summary visualization
    """
    # Calculate metrics for prediction period
    prediction_start = pd.to_datetime('2024-07-11')
    pred_df = df[df['date'] >= prediction_start].copy()
    pred_columns = [col for col in df.columns if col.startswith('pred_')]
    
    # Collect metrics
    results = []
    
    for col in pred_columns:
        metrics = calculate_metrics(pred_df, col)
        if not np.isnan(metrics['RMSE']):
            model_name = col.replace('pred_', '').replace('_', ' ').title()
            results.append({
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2']
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. RMSE ranking
    # Create gradient from green (best) to red (worst)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results_df)))
    bars = ax1.barh(results_df['Model'], results_df['RMSE'], color=colors, edgecolor='black', linewidth=1)
    ax1.set_xlabel('RMSE ($)', fontsize=12)
    ax1.set_title('Model Ranking by RMSE (Lower is Better)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'${width:.2f}', ha='left', va='center', fontsize=10)
    
    # 2. Best model vs actual (last 3 months)
    best_model = results_df.iloc[0]['Model'].lower().replace(' ', '_')
    best_model_col = f'pred_{best_model}'
    
    last_3_months = pred_df[pred_df['date'] >= pred_df['date'].max() - pd.Timedelta(days=90)]
    
    ax2.plot(last_3_months['date'], last_3_months['actual_close'], 'k-', 
             linewidth=3, label='Actual Price', alpha=0.9)
    
    if best_model_col in last_3_months.columns:
        ax2.plot(last_3_months['date'], last_3_months[best_model_col], '-', 
                color='#FF0000', linewidth=2.5, label=f'Best Model ({results_df.iloc[0]["Model"]})', alpha=0.8)
    
    ax2.set_title('Best Model Performance (Last 3 Months)', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Stock Price ($)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/performance_summary.png")
    plt.close()
    
    return results_df

def create_correlation_heatmap(df):
    """
    Create correlation heatmap between predictions
    """
    # Get prediction columns
    pred_columns = [col for col in df.columns if col.startswith('pred_')]
    
    # Filter for prediction period
    prediction_start = pd.to_datetime('2024-07-11')
    pred_df = df[df['date'] >= prediction_start].copy()
    
    # Select only prediction columns and actual
    corr_columns = ['actual_close'] + pred_columns
    corr_data = pred_df[corr_columns].copy()
    
    # Rename columns for better display
    rename_dict = {'actual_close': 'Actual'}
    for col in pred_columns:
        model_name = col.replace('pred_', '').replace('_', ' ').title()
        rename_dict[col] = model_name
    
    corr_data.rename(columns=rename_dict, inplace=True)
    
    # Calculate correlation
    correlation = corr_data.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.3f', 
                cmap='RdBu_r', center=1.0, square=True,
                linewidths=1, cbar_kws={"shrink": .8},
                vmin=0.9, vmax=1.0)
    
    plt.title('Correlation Between Actual and Predicted Prices', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: output/correlation_heatmap.png")
    plt.close()

def save_consolidated_data(df, results_df):
    """
    Save consolidated data and results to CSV
    """
    # Save consolidated predictions
    cols = ['date', 'actual_close']
    pred_cols = sorted([col for col in df.columns if col.startswith('pred_')])
    cols.extend(pred_cols)
    
    export_df = df[cols].copy()
    export_df.to_csv('output/consolidated_predictions.csv', index=False)
    print("✓ Saved: output/consolidated_predictions.csv")
    
    # Save prediction period only
    pred_start = pd.to_datetime('2024-07-11')
    pred_only = export_df[export_df['date'] >= pred_start].copy()
    pred_only.to_csv('output/predictions_only.csv', index=False)
    print("✓ Saved: output/predictions_only.csv")
    
    # Save metrics
    results_df.to_csv('output/model_performance_metrics.csv', index=False)
    print("✓ Saved: output/model_performance_metrics.csv")

# Main execution
if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("NVIDIA STOCK PREDICTION ANALYSIS")
        print("="*60)
        
        # Load data
        print("\nLoading and consolidating data...")
        df = load_and_process_data('data/')
        
        print(f"\nTotal rows: {len(df)}")
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        print("\n1. Main comparison charts...")
        create_main_visualization(df)
        
        print("\n2. Error analysis...")
        create_error_analysis(df)
        
        print("\n3. Performance summary...")
        results_df = create_performance_summary(df)
        
        print("\n4. Correlation heatmap...")
        create_correlation_heatmap(df)
        
        # Save data
        print("\n5. Saving consolidated data...")
        save_consolidated_data(df, results_df)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nAll files saved in 'output' directory:")
        print("\nVisualizations (PNG):")
        print("  • output/stock_predictions_comparison.png - Main comparison")
        print("  • output/error_analysis.png - Error analysis")
        print("  • output/performance_summary.png - Performance summary")
        print("  • output/correlation_heatmap.png - Correlation analysis")
        print("\nData files (CSV):")
        print("  • output/consolidated_predictions.csv - All data")
        print("  • output/predictions_only.csv - Predictions only")
        print("  • output/model_performance_metrics.csv - Performance metrics")
        
        print("\n✅ All files generated successfully in 'output' directory!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()