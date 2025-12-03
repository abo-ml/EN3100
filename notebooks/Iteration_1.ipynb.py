"""
# Iteration ${i} Walkthrough Notebook (Python script surrogate)

This notebook-style script guides the reader through the objectives, methodology, and example code snippets for Iteration ${i}. Convert to a true `.ipynb` if interactive exploration is preferred.
"""

# %% [markdown]
# ## Setup
# Import required packages and ensure the project root is on the path.

# %%
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ITERATION_ID = pathlib.Path(__file__).stem.split("_")[1]
MODULE_SUFFIX = {
    "1": "baseline",
    "2": "ensemble",
    "3": "lstm",
    "4": "transformer",
    "5": "meta_ensemble",
}.get(ITERATION_ID, "baseline")

# %% [markdown]
# ## Objectives
# - Review the engineered dataset.
# - Demonstrate how to invoke the Iteration ${i} training script.
# - Inspect metrics saved in `reports/`.

# %%
print(f"Run the corresponding model script via: python -m src.models.iteration{ITERATION_ID}_{MODULE_SUFFIX}")

# %% [markdown]
# ## Data Preview
# Uncomment the following cells after generating `model_features.parquet` to inspect the data.

# %%
# import pandas as pd
# features = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "model_features.parquet")
# display(features.head())

# %% [markdown]
# ## Notes
# - Extend this notebook with exploratory plots, diagnostics, and narrative commentary.
# - Embed comparisons between walk-forward splits and out-of-sample performance.
# - Document insights for the dissertation write-up.
