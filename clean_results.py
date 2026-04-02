import pandas as pd

def clean_csv():
    path = 'results/metrics/evaluation_summary.csv'
    df = pd.read_csv(path)
    
    # We want to keep ONLY the most recent entry for each unique model
    # (Since our best results are at the bottom for each)
    df_cleaned = df.drop_duplicates(subset=['Model'], keep='last')
    
    df_cleaned.to_csv(path, index=False)
    print(f"Cleaned {path}. Kept final results for each model.")

if __name__ == '__main__':
    clean_csv()
