import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load the data
    # Make sure the CSV file name matches exactly where you saved it
    csv_file = r'D:\Projects\osim-rl\all_stiffness_rewards.csv' 
    try:
        df = pd.read_csv(csv_file)
        print("Successfully loaded data!")
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Make sure it is in the same folder.")
        return

    # 2. Calculate summary statistics per stiffness
    summary = df.groupby('Stiffness').agg(
        total_steps=('Step', 'count'),
        max_step=('Step', 'max'),
        mean_reward=('Reward', 'mean'),
        sum_reward=('Reward', 'sum'),
        median_reward=('Reward', 'median'),
        std_reward=('Reward', 'std')
    ).reset_index()

    print("\n--- Summary Statistics ---")
    print(summary.to_string(index=False))

    # 3. Set the visual style for the plots
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: Reward vs Step for each stiffness ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Step', y='Reward', hue='Stiffness', palette='tab10')
    plt.title('Reward over Time (Steps) for Different Stiffness Values', fontsize=14)
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(title='Ankle Stiffness')
    plt.tight_layout()
    plt.savefig('reward_vs_step.png', dpi=300)
    plt.show() # This will pop up the window for you to see

    # --- PLOT 2: Total Reward per Stiffness ---
    plt.figure(figsize=(8, 5))
    # Convert Stiffness to string so seaborn treats it as a category for the bar chart
    summary['Stiffness_cat'] = summary['Stiffness'].astype(str)
    
    sns.barplot(data=summary, x='Stiffness_cat', y='sum_reward', hue='Stiffness_cat', palette='viridis', legend=False)
    plt.title('Total Accumulated Reward by Ankle Stiffness', fontsize=14)
    plt.xlabel('Ankle Stiffness', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.tight_layout()
    plt.savefig('total_reward.png', dpi=300)
    plt.show()

    # --- PLOT 3: Mean Reward per Stiffness ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x='Stiffness_cat', y='mean_reward', hue='Stiffness_cat', palette='magma', legend=False)
    plt.title('Mean Reward per Step by Ankle Stiffness', fontsize=14)
    plt.xlabel('Ankle Stiffness', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    # Set y-axis slightly zoomed in to highlight differences
    plt.ylim(0.7, 0.85) 
    plt.tight_layout()
    plt.savefig('mean_reward.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()