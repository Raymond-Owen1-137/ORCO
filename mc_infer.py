import pandas as pd
import torch
import joblib
import numpy as np
from train import ORCO_Net
import matplotlib.pyplot as plt

residue_letters = {
    0:'A', 1:'R', 2:'N', 3:'D', 4:'C', 5:'E', 6:'Q', 7:'G',
    8:'H', 9:'I', 10:'L', 11:'K', 12:'M', 13:'F', 14:'P',
    15:'S', 16:'T', 17:'W', 18:'Y', 19:'V'
}

def mc_predict(model, x, num_samples=100):
    model.train()
    preds = torch.stack([model(x) for _ in range(num_samples)])
    return preds.mean(dim=0), preds.std(dim=0)

def main():
    df = pd.read_csv("bmrb_4769_labeled.csv")
    X = df[['CA', 'CB']].values.astype('float32')
    y_true = df['label'].values.astype('int64')

    scaler = joblib.load("orco_scaler.pkl")
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = ORCO_Net()
    model.load_state_dict(torch.load("orco_model.pt"))

    mean_log_probs, std_probs = mc_predict(model, X_tensor, num_samples=100)
    mean_probs = torch.exp(mean_log_probs)
    top_probs, top_indices = torch.topk(mean_probs, k=5, dim=1)

    rows = []
    for i in range(len(X)):
        topk_preds = [top_indices[i][k].item() for k in range(5)]
        topk_probs = [top_probs[i][k].item() for k in range(5)]
        topk_stds = [std_probs[i][top_indices[i][k]].item() for k in range(5)]
        top1_label = topk_preds[0]
        true_label = y_true[i]
        accepted = topk_probs[0] > 0.7 and topk_stds[0] < 0.5
        narrow = sum(p > 0.2 for p in topk_probs) <= 3
        rows.append({
            "index": i+1,
            "CA": X[i][0],
            "CB": X[i][1],
            "true_label": true_label,
            "true_residue": residue_letters[true_label],
            "pred_label": top1_label,
            "pred_residue": residue_letters[top1_label],
            "top1_prob": topk_probs[0],
            "top1_std": topk_stds[0],
            "accepted": accepted,
            "narrowed": narrow,
            "top_k_count_above_0.2": sum(p > 0.2 for p in topk_probs)
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv("bmrb_4769_mc_results.csv", index=False)
    print("✅ Results saved to bmrb_4769_mc_results.csv")

if __name__ == "__main__":
    main()