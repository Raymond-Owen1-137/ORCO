import matplotlib.pyplot as plt
import pandas as pd

residue_letters = {
    0:'A', 1:'R', 2:'N', 3:'D', 4:'C', 5:'E', 6:'Q', 7:'G',
    8:'H', 9:'I', 10:'L', 11:'K', 12:'M', 13:'F', 14:'P',
    15:'S', 16:'T', 17:'W', 18:'Y', 19:'V'
}

def plot_results(csv_file="bmrb_4769_mc_results.csv", output_file="mc_results_plot.png"):
    df = pd.read_csv(csv_file)

    n_total = len(df)
    n_accepted = df['accepted'].sum()
    n_correct = (df['true_label'] == df['pred_label']).sum()
    n_narrowed = df['narrowed'].sum()

    plt.figure(figsize=(14, 6))
    for i, row in df.iterrows():
        idx = int(row['index'])
        y = row['pred_label']
        yerr = row['top1_std']
        pred = residue_letters[y]
        true = residue_letters[row['true_label']]
        color = 'green' if row['accepted'] and pred == true else 'orange' if row['accepted'] else 'red'
        marker = 'o' if row['accepted'] and pred == true else '^' if row['accepted'] else 'x'
        plt.errorbar(idx, y, yerr=yerr, fmt=marker, color=color, capsize=3)
        label = f"{pred}/{true}" if pred != true else pred
        plt.text(idx, y + 0.5, label, ha='center', fontsize=8, fontweight='bold' if pred != true else 'normal')

    plt.xticks(range(1, n_total + 1))
    plt.yticks(list(residue_letters.keys()), list(residue_letters.values()))
    plt.xlabel("Residue Index")
    plt.ylabel("Predicted Residue")
    plt.title("Residue Prediction Results with Uncertainty")

    summary = f"Accepted: {n_accepted}/{n_total} | Correct Top-1: {n_correct}/{n_total} | Narrowed: {n_narrowed}/{n_total}"
    plt.text(0.5, 1.05, summary, ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_results()