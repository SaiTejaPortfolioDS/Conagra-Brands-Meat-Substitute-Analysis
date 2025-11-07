# Conagra Meat Substitute — Market Analysis (Lasso/Elastic Net)

A concise, portfolio-ready analysis of Conagra’s **Meat Substitutes** category using **regularized linear models** to explain and predict revenue drivers. The notebook focuses on building an interpretable pipeline (scaling → regularized regression) over a one‑hot encoded dataset with category/product features.

## 🧠 Overview
- **Goal:** Identify key drivers of sales/performance in the Meat Substitutes category and quantify their impact.
- **Approach:** Exploratory analysis followed by **Lasso** and **Elastic Net** regression in a scikit‑learn `Pipeline` with cross‑validated alpha selection.
- **Outputs:** 
  - Cross‑validated best `alpha` (regularization strength)
  - Test set **R²** score
  - Ranked feature coefficients for interpretability (top positive/negative drivers)

## 📊 Data
The notebook references the following files (place them under `data/` in your repo):
- `Final Data.csv` — consolidated modeling table (numeric + encoded features)
- `One_Hot_Encoded_File.csv` — base one‑hot encoded data
- `One_Hot_Encoded_File_with_Interactions.csv` — optional interaction‑augmented data
- `Top_5_Meat_Substitute_With_Category_Info.xlsx` — reference/lookup for top SKUs & categories

> Notes: Columns in code indicate features like `total_ounces`, `unit_*`, `sub_category_*`, and seasonal/promotional flags (e.g., `season_promo`).

## 🧮 Methodology
- **Preprocessing:** `StandardScaler` applied within the modeling pipeline so scaling is learned only on training data.
- **Models:** `Lasso` (with CV) and `ElasticNet` (optional), tuned over a grid of alphas (e.g., `n_alphas=100`, range ~[0, 1]).
- **Evaluation:** Train/test split (80/20). Primary metric: **R²** (`sklearn.metrics.r2_score`).
- **Interpretability:** Sorted absolute coefficients to surface strongest drivers (positive/negative).

## 🧰 Tech Stack
- **Python:** `pandas`, `numpy`
- **Modeling:** `scikit-learn` (`Pipeline`, `StandardScaler`, `Lasso`, `ElasticNet`, `train_test_split`, `r2_score`)
- **Viz:** `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook (`Final_code.ipynb`)

## 📂 Project Structure
```
conagra-meat-substitute-analysis/
├─ data/
│  ├─ Final Data.csv
│  ├─ One_Hot_Encoded_File.csv
│  ├─ One_Hot_Encoded_File_with_Interactions.csv
│  └─ Top_5_Meat_Substitute_With_Category_Info.xlsx
├─ notebooks/
│  └─ Final_code.ipynb
├─ README.md
└─ requirements.txt
```

## ⚙️ Setup & Usage
```bash
# 1) Clone
git clone https://github.com/<your-username>/conagra-meat-substitute-analysis.git
cd conagra-meat-substitute-analysis

# 2) Create env (optional)
python -m venv .venv && source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate                            # Windows

# 3) Install deps
pip install -r requirements.txt

# 4) Add data files to ./data (see list above)

# 5) Run notebook
jupyter notebook notebooks/Final_code.ipynb
```

### `requirements.txt`
If you don’t already have one, use this baseline:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## 🔎 Reproducing Key Results
Inside the notebook you’ll find cells that:
- Fit the pipeline: `StandardScaler` → `Lasso` (or `ElasticNet`)
- Print best alpha: `pipeline.named_steps['lasso'].alpha_`
- Score on hold‑out: `r2_score(y_test, y_pred)`
- Extract coefficients: `pipeline.named_steps['lasso'].coef_` and rank by magnitude

> Tip: If using the interactions file, re‑point the data loading cell to `One_Hot_Encoded_File_with_Interactions.csv` and re‑run to compare performance & coefficient stability.

## 💡 Insights & Business Framing (Template)
- **Promotion & Seasonality:** Quantify lift from promotional periods (`season_promo`) and seasonal effects.
- **Pack Size / Ounces:** Assess elasticity around `total_ounces` or `unit_*` attributes.
- **Sub‑categories / Brands:** Identify sub‑categories (`sub_category_*`) with outsized positive or negative coefficients.
- **Actionable Recommendations:** Price/pack architecture optimization, promo calendar planning, and assortment prioritization for top SKUs.

## 🤝 Author
**Sai Teja KMVP**  
Master’s in Business Analytics & AI, UTD  
GitHub: https://github.com/<your-username> · LinkedIn: https://www.linkedin.com/in/<your-handle>/

---

> **Note:** This README was generated from the structure and code found in `Final_code.ipynb`. Populate the exact **R²** and **best alpha** after running with your local `data/` files.
