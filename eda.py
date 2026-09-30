import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_PATH = "creditcard.csv"
OUTPUT_DIR = "eda_outputs"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: Load Data")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

print("\n" + "=" * 70)
print("STEP 2: Basic Info & Data Types")
print("=" * 70)
print("\nData Types:")
print(df.dtypes)

print("\nMemory Usage:")
print(df.memory_usage(deep=True).sort_values(ascending=False).head(10))

print("\n" + "=" * 70)
print("STEP 3: Statistical Summary (numerical)")
print("=" * 70)
print(df.describe().T)

print("\n" + "=" * 70)
print("STEP 4: Missing Values")
print("=" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(3)
missing_df = pd.DataFrame({"Missing": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values(by="Missing", ascending=False)
if missing_df.empty:
    print("No missing values found in the dataset.")
else:
    print("Columns with missing values:")
    print(missing_df)

print("\n" + "=" * 70)
print("STEP 5: Target Variable Analysis (Class)")
print("=" * 70)
class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100
target_df = pd.DataFrame({"Count": class_counts, "Percentage": class_pct.round(4)})
print("Class distribution:")
print(target_df)
print(f"\nClass balance ratio: {class_counts[0]} : {class_counts[1]} = {class_counts[0]/class_counts[1]:.1f}:1 (legit:fraud)")
print(f"Fraud rate: {class_pct[1]:.4f}%")

print("\n" + "=" * 70)
print("STEP 6: Duplicate Rows")
print("=" * 70)
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")
if dup_count > 0:
    print("Removing duplicates...")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dataset after removing duplicates: {df.shape[0]} rows")

print("\n" + "=" * 70)
print("STEP 7: Feature Distributions & Visualizations")
print("=" * 70)

# Class distribution plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Class", data=df, ax=ax, hue="Class", palette="Set2", legend=False)
ax.set_title("Class Distribution (0 = Legit, 1 = Fraud)")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
plt.close()
print("Saved: class_distribution.png")

# Time distribution
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Time"], bins=100, ax=ax, color="steelblue")
ax.set_title("Time Distribution (seconds from first transaction)")
ax.set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/time_distribution.png", dpi=150)
plt.close()
print("Saved: time_distribution.png")

# Amount distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df["Amount"], bins=100, ax=axes[0], color="darkorange")
axes[0].set_title("Transaction Amount Distribution")
axes[0].set_xlabel("Amount")

# Log amount
df_log = df.copy()
df_log["Amount_log"] = np.log1p(df_log["Amount"])
sns.histplot(df_log["Amount_log"], bins=100, ax=axes[1], color="seagreen")
axes[1].set_title("Transaction Amount (Log1p)")
axes[1].set_xlabel("log1p(Amount)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/amount_distribution.png", dpi=150)
plt.close()
print("Saved: amount_distribution.png")

# Amount vs Class
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x="Class", y="Amount", hue="Class", data=df, ax=ax, palette="Set2", legend=False)
ax.set_yscale("log")
ax.set_title("Amount by Class (log scale)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/amount_by_class.png", dpi=150)
plt.close()
print("Saved: amount_by_class.png")

# Correlation heatmap for V features + Amount + Class
v_cols = [c for c in df.columns if c.startswith("V")]
corr_cols = v_cols + ["Amount", "Class"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar=True,
            square=True, linewidths=0.4, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap (V features + Amount + Class)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# Correlations with Class
class_corr = corr["Class"].drop("Class").sort_values(ascending=False)
print("\nCorrelations of features with Class (sorted):")
print(class_corr.round(4))

# Save correlation with class to CSV
class_corr.round(4).to_csv(f"{OUTPUT_DIR}/class_correlations.csv", header=["Correlation"])

# V features distributions (sample)
v_features = [f"V{i}" for i in range(1, 29)]
fig, axes = plt.subplots(7, 4, figsize=(18, 22))
axes = axes.flatten()
for i, col in enumerate(v_features):
    sns.histplot(df[col], bins=50, ax=axes[i], color="steelblue", alpha=0.7)
    axes[i].set_title(col, fontsize=8)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
plt.suptitle("Distributions of V1-V28 Features", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/v_features_distributions.png", dpi=150)
plt.close()
print("Saved: v_features_distributions.png")

# V features by Class (boxplots for top correlated features)
top_pos = class_corr.head(6).index.tolist()
top_neg = class_corr.tail(6).index.tolist()
top_features = top_pos + top_neg

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(top_features):
    sns.boxplot(x="Class", y=col, hue="Class", data=df, ax=axes[i], palette="Set2", legend=False)
    axes[i].set_title(f"{col} by Class")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_correlated_by_class.png", dpi=150)
plt.close()
print("Saved: top_correlated_by_class.png")

print("\n" + "=" * 70)
print("STEP 8: Summary Statistics by Class")
print("=" * 70)
for col in ["Time", "Amount"] + [f"V{i}" for i in range(1, 6)]:
    print(f"\n{col}:")
    print(df.groupby("Class")[col].describe().round(3))

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.4f}%)")
print(f"Legit cases: {(1-df['Class']).sum()} ({(1-df['Class']).sum()/len(df)*100:.4f}%)")
print(f"Duplicate rows removed: {dup_count}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Output files saved in: {OUTPUT_DIR}/")
