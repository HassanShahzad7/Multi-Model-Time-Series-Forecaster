import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import os

def get_nvda_stock_data(start_date="2022-01-01", end_date="2025-12-31"):
    """Get NVDA stock data from Yahoo Finance"""
    print(f"📈 Getting NVDA stock data from {start_date} to {end_date}...")
    
    nvda = yf.Ticker("NVDA")
    stock_data = nvda.history(start=start_date, end=end_date)
    
    # Convert to DataFrame with date as column
    stock_df = stock_data.reset_index()
    stock_df['date'] = stock_df['Date'].dt.date
    stock_df = stock_df.drop('Date', axis=1)
    
    # Rename columns to lowercase
    stock_df.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'date']
    
    # Reorder columns
    stock_df = stock_df[['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']]
    
    print(f"✅ Retrieved {len(stock_df)} trading days")
    return stock_df

def get_nvda_news_alpha_vantage(api_key, start_date, end_date):
    """Get NVDA news from Alpha Vantage for date range"""
    
    # Convert dates to Alpha Vantage format (YYYYMMDDTHHMM)
    start_formatted = start_date.replace('-', '') + 'T0000'
    end_formatted = end_date.replace('-', '') + 'T2359'
    
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': 'NVDA',
        'time_from': start_formatted,
        'time_to': end_formatted,
        'limit': 1000,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for errors
        if 'Error Message' in data:
            print(f"  ❌ Error: {data['Error Message']}")
            return []
        
        if 'Note' in data:
            print(f"  ⚠️  API Limit: {data['Note']}")
            return []
            
        # Extract news articles
        articles = []
        if 'feed' in data:
            for article in data['feed']:
                try:
                    # Parse date
                    time_published = article['time_published']
                    pub_date = datetime.strptime(time_published[:8], '%Y%m%d').date()
                    
                    articles.append({
                        'date': pub_date,
                        'title': article['title'],
                        'summary': article.get('summary', ''),
                        'sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'sentiment_label': article.get('overall_sentiment_label', ''),
                        'source': article.get('source', ''),
                        'url': article.get('url', '')
                    })
                except Exception as e:
                    continue
        
        return articles
        
    except Exception as e:
        print(f"  ❌ Request error: {e}")
        return []

def collect_nvda_news_by_year(api_key, start_year=2022, end_year=2025):
    """Collect NVDA news year by year"""
    print(f"📰 Getting NVDA news from {start_year} to {end_year}...")
    
    all_articles = []
    
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        print(f"   📅 {year}... ", end="")
        
        year_articles = get_nvda_news_alpha_vantage(api_key, start_date, end_date)
        
        if year_articles:
            all_articles.extend(year_articles)
            print(f"✅ {len(year_articles)} articles")
        else:
            print(f"❌ Failed")
        
        # Delay between requests
        if year < end_year:
            time.sleep(2)
    
    # Convert to DataFrame and get one news per day
    if all_articles:
        news_df = pd.DataFrame(all_articles)
        
        # For each date, keep the article with highest absolute sentiment score
        news_df['abs_sentiment'] = abs(news_df['sentiment_score'])
        daily_news = news_df.loc[news_df.groupby('date')['abs_sentiment'].idxmax()].copy()
        daily_news = daily_news.drop('abs_sentiment', axis=1)
        daily_news = daily_news.sort_values('date').reset_index(drop=True)
        
        print(f"✅ Processed to {len(daily_news)} daily news articles")
        return daily_news
    
    return pd.DataFrame()

def create_combined_nvda_dataset(api_key, start_date="2022-01-01", end_date="2025-12-31"):
    """Create combined NVDA dataset with stock prices and news"""
    
    combined_filename = 'nvda_combined_dataset_2022_2025.csv'
    
    # Check if dataset already exists
    if os.path.exists(combined_filename):
        print(f"📁 Found existing combined dataset: {combined_filename}")
        response = input("Load existing data? (y/n): ").lower()
        if response == 'y':
            df = pd.read_csv(combined_filename)
            df['date'] = pd.to_datetime(df['date']).dt.date
            print(f"✅ Loaded {len(df)} rows from existing dataset")
            return df
    
    print(f"🚀 Creating combined NVDA dataset ({start_date} to {end_date})")
    
    # 1. Get stock data
    stock_df = get_nvda_stock_data(start_date, end_date)
    
    # 2. Get news data
    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    news_df = collect_nvda_news_by_year(api_key, start_year, end_year)
    
    # 3. Merge stock and news data
    print(f"\n🔗 Combining stock and news data...")
    
    # Merge on date (left join to keep all trading days)
    combined_df = stock_df.merge(news_df, on='date', how='left')
    
    # Fill missing news with default values
    combined_df['title'] = combined_df['title'].fillna(f'NVDA trading session')
    combined_df['summary'] = combined_df['summary'].fillna('No significant news for NVIDIA today')
    combined_df['sentiment_score'] = combined_df['sentiment_score'].fillna(0.0)
    combined_df['sentiment_label'] = combined_df['sentiment_label'].fillna('Neutral')
    combined_df['source'] = combined_df['source'].fillna('Generated')
    combined_df['url'] = combined_df['url'].fillna('')
    
    # Create a text field combining title and summary for MoAT model
    combined_df['text'] = combined_df['title'] + '. ' + combined_df['summary']
    
    # Reorder columns for clarity
    columns_order = [
        'date', 
        # Stock data
        'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits',
        # News data
        'title', 'summary', 'text', 'sentiment_score', 'sentiment_label', 'source', 'url'
    ]
    combined_df = combined_df[columns_order]
    
    # Sort by date
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Save combined dataset
    combined_df.to_csv(combined_filename, index=False)
    print(f"💾 Combined dataset saved to: {combined_filename}")
    
    # Show statistics
    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   • Total rows: {len(combined_df)}")
    print(f"   • Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"   • Trading days: {len(combined_df)}")
    print(f"   • Days with real news: {(combined_df['source'] != 'Generated').sum()}")
    print(f"   • Days with generated news: {(combined_df['source'] == 'Generated').sum()}")
    print(f"   • Average stock price: ${combined_df['close'].mean():.2f}")
    print(f"   • Price range: ${combined_df['close'].min():.2f} - ${combined_df['close'].max():.2f}")
    print(f"   • Sentiment distribution: {combined_df['sentiment_label'].value_counts().to_dict()}")
    
    return combined_df

def show_sample_combined_data(df, n=5):
    """Show sample rows from combined dataset"""
    print(f"\n📋 Sample Combined Data (showing {n} rows):")
    
    sample = df.sample(n=min(n, len(df)))
    
    for idx, row in sample.iterrows():
        print(f"\n--- Row #{idx} ---")
        print(f"Date: {row['date']}")
        print(f"Stock Close: ${row['close']:.2f}")
        print(f"Volume: {row['volume']:,}")
        print(f"News Title: {row['title'][:60]}...")
        print(f"News Summary: {row['summary'][:80]}...")
        print(f"Sentiment: {row['sentiment_label']} ({row['sentiment_score']:.3f})")
        print(f"Source: {row['source']}")

# Main execution
def main():
    # Your Alpha Vantage API key
    API_KEY = "O94XFFNBMDHDDZJY"
    
    # Create combined dataset for 2022-2025
    combined_df = create_combined_nvda_dataset(
        api_key=API_KEY, 
        start_date="2022-01-01", 
        end_date="2025-12-31"
    )
    
    if combined_df is not None and len(combined_df) > 0:
        # Show sample data
        show_sample_combined_data(combined_df)
        
        print(f"\n✅ Combined dataset creation complete!")
        print(f"   📄 File: nvda_combined_dataset_2022_2025.csv")
        print(f"   📊 Rows: {len(combined_df)}")
        print(f"   🏛️  Columns: {len(combined_df.columns)}")
        print(f"\n🔧 Ready for MoAT model training!")
        
        # Show column info
        print(f"\n📋 Dataset Columns:")
        for i, col in enumerate(combined_df.columns, 1):
            print(f"   {i:2d}. {col}")
    
    else:
        print("❌ Failed to create combined dataset!")

if __name__ == "__main__":
    main()