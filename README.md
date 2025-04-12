# OpenSource Residue Classification Oracle (ORCO)

ORCO is an open-source neural network system for classifying amino acid residue types from NMR chemical shift data. It is designed to support automated backbone assignment workflows using Cα and Cβ shifts and is suitable for exploratory modeling, reproducibility testing, or integration into larger pipelines.

---

## 🚀 What It Does

ORCO predicts the most likely amino acid residue based on **Cα and Cβ chemical shifts** using a neural network trained on curated BMRB data. It includes:

- Data preparation from BMRB JSON datasets
- Neural network training
- Monte Carlo Dropout inference with uncertainty estimation
- CSV result export and annotated visualizations

---

## 📦 What's Included

| File                     | Purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| `load_json.py`           | Load a BMRB entry, create `.fasta` and labeled `.csv`     |
| `train.py`               | Train a PyTorch model on CA/CB data                       |
| `mc_infer.py`            | Perform Monte Carlo inference with uncertainty            |
| `plot_mc_results.py`     | Visualize predicted vs true residues with uncertainty     |

---

## 🧪 Sample Workflow

1. Load and process a BMRB entry:
```bash
python load_json.py 4769
```

2. Train a model:
```bash
python train.py
```

3. Run inference:
```bash
python mc_infer.py
```

4. Plot annotated results:
```bash
python plot_mc_results.py
```

---

## 📂 Input Format

The model expects a CSV with:
```
CA,CB,label
56.1,30.5,0
...
```
Residues are labeled using standard 0–19 indexing.

---

## 🎯 Output

- `orco_model.pt` — trained model weights
- `orco_scaler.pkl` — feature scaler
- `bmrb_XXXX_mc_results.csv` — predictions, uncertainty, acceptance
- `mc_results_plot.png` — annotated visualization

---

## 🤝 License

This project is open-source and intended for educational and research use. For integration into production pipelines, please verify prediction accuracy under your own validation framework.

---

## 👨‍🔬 About

Developed by a physics undergraduate exploring machine learning applications in structural biology and NMR spectroscopy. Built for clarity, reproducibility, and future expansion into semi-automated assignment tools.