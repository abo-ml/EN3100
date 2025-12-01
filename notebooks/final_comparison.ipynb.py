# Notebook: final_comparison.ipynb
# Purpose: Aggregate iteration metrics, visualise RMSE/MAE/R2 across models,
# and provide a quick textual summary of the best-performing approach.
#
# Usage (Colab/local):
#   %cd "/content/EN3100 Repo"  # adjust path as needed
#   %run notebooks/final_comparison.ipynb.py
#
# Requirements: matplotlib, pandas installed via requirements.txt.

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

# ---------------------------------------------------------------------------
# Load metrics from markdown reports
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
reports_dir = repo_root / "reports"
report_files = sorted(reports_dir.glob("iteration_*_results.md"))

rows = []
for path in report_files:
    text = path.read_text()
    rmse = re.search(r"RMSE[: ]+([\d\.]+)", text)
    mae = re.search(r"MAE[: ]+([\d\.]+)", text)
    r2 = re.search(r"R2[: ]+([\d\.\-]+)", text)
    rows.append(
        {
            "iteration": path.stem,
            "RMSE": float(rmse.group(1)) if rmse else None,
            "MAE": float(mae.group(1)) if mae else None,
            "R2": float(r2.group(1)) if r2 else None,
        }
    )

df = pd.DataFrame(rows).set_index("iteration")
display(Markdown("### Aggregated metrics"))
display(df)

# ---------------------------------------------------------------------------
# Plot metrics by iteration
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
    df[metric].plot(kind="bar", ax=ax, title=metric)
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
fig_path = reports_dir / "figures" / "iteration_comparison.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path)
print(f"Saved metric comparison plot to {fig_path}")

# ---------------------------------------------------------------------------
# Identify the best model(s)
# ---------------------------------------------------------------------------
best_rmse = df["RMSE"].idxmin() if df["RMSE"].notna().any() else "N/A"
best_mae = df["MAE"].idxmin() if df["MAE"].notna().any() else "N/A"
best_r2 = df["R2"].idxmax() if df["R2"].notna().any() else "N/A"

summary = (
    f"* Lowest RMSE: **{best_rmse}**\n"
    f"* Lowest MAE: **{best_mae}**\n"
    f"* Highest R²: **{best_r2}**\n"
)
display(Markdown("### Model comparison summary"))
display(Markdown(summary))
