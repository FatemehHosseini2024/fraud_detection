import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pipeline import prepare_data

RANDOM_STATE = 42

print("=" * 70)
print("RANDOM FOREST TRAINING TIME MEASUREMENT")
print("=" * 70)

# Load and prepare data (same as real pipeline)
df, train, test, _ = prepare_data()
X = train.drop(columns=["Class"])
y = train["Class"]

print(f"Full train set: {X.shape[0]} rows, {X.shape[1]} features")

# Simulate inner fold size: 80% train * 4/5 = 64% of full
X_outer_train, _, y_outer_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_inner_train, _, y_inner_train, _ = train_test_split(
    X_outer_train, y_outer_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_outer_train
)

print(f"Inner training size: {X_inner_train.shape[0]} rows (~64% of full)")
print(f"Fraud rate: {y_inner_train.sum()/len(y_inner_train)*100:.4f}%")

# Heavy configuration (ceiling)
heavy_params = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "max_samples": None,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Light configuration (floor)
light_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "max_samples": 0.8,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

def time_training(params, label, n_repeats=3):
    print(f"\n--- {label} ---")
    print(f"Params: {params}")
    times = []
    
    for i in range(n_repeats):
        rf = RandomForestClassifier(**params)
        start = time.perf_counter()
        rf.fit(X_inner_train, y_inner_train)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} seconds")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} seconds")
    return avg_time

# Warm-up run (first run often slower)
print("\n--- Warm-up run (light) ---")
rf_warm = RandomForestClassifier(**light_params)
_ = time.perf_counter()
rf_warm.fit(X_inner_train, y_inner_train)
print(f"  Done")

# Measure heavy and light
heavy_avg = time_training(heavy_params, "HEAVY (ceiling)")
light_avg = time_training(light_params, "LIGHT (floor)")

# Estimate total for nested CV search
# Example: 5 outer x 5 inner x 30 trials = 750 trainings
# Plus final models: 5 outer final fits
n_trainings = 5 * 5 * 30 + 5  # 755

print("\n" + "=" * 70)
print("ESTIMATED TOTAL TIME (5 outer x 5 inner x 30 trials + 5 final)")
print("=" * 70)
print(f"Heavy per training:  {heavy_avg:.1f}s")
print(f"Light per training:  {light_avg:.1f}s")
print(f"Midpoint:            {(heavy_avg + light_avg)/2:.1f}s")
print(f"\nHeavy total:  {heavy_avg * n_trainings / 3600:.1f} hours")
print(f"Light total:  {light_avg * n_trainings / 3600:.1f} hours")
print(f"Midpoint:     {(heavy_avg + light_avg)/2 * n_trainings / 3600:.1f} hours")

print("\nDone.")