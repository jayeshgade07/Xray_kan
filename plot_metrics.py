import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_training_curves():
    os.makedirs('results/plots', exist_ok=True)
    models = ['cnn', 'dense', 'kan']
    
    plt.figure(figsize=(10, 6))
    
    for model in models:
        csv_path = f'results/metrics/{model}_training_stats.csv'
        if not os.path.exists(csv_path):
            print(f"Skipping {model}: {csv_path} not found.")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Plot Validation Loss for comparison
        plt.plot(df['Epoch'], df['Val_Loss'], marker='o', label=f'{model.upper()} Val Loss')
        
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/val_loss_comparison.png')
    print("Saved results/plots/val_loss_comparison.png")
    
def plot_evaluation_metrics():
    metrics_path = 'results/metrics/evaluation_summary.csv'
    if not os.path.exists(metrics_path):
        print(f"{metrics_path} not found.")
        return
        
    df = pd.read_csv(metrics_path)
    
    # We want to plot AUC and F1 Score for the models
    df_melted = df.melt(id_vars='Model', value_vars=['AUC', 'F1', 'Accuracy'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model')
    plt.title('Final Evaluation Metrics Comparison')
    plt.ylim(0, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('results/plots/evaluation_metrics.png')
    print("Saved results/plots/evaluation_metrics.png")

if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    plot_training_curves()
    plot_evaluation_metrics()
