import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import gzip
import urllib.request
import gc
import time

np.random.seed(42)

# ====================================================================
# 1. DOWNLOAD FULL KDD CUP 99 + TEST SET
# ====================================================================
DATA_DIR = './kddcup99_data'
os.makedirs(DATA_DIR, exist_ok=True)

files = {
    'train_full': {
        'url': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz',
        'gz': os.path.join(DATA_DIR, 'kddcup.data.gz'),
        'csv': os.path.join(DATA_DIR, 'kddcup_full.csv'),
    },
    'test': {
        'url': 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz',
        'gz': os.path.join(DATA_DIR, 'corrected.gz'),
        'csv': os.path.join(DATA_DIR, 'kddcup_test.csv'),
    }
}

col_names = [
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

categorical_cols = ['protocol_type', 'service', 'flag']

print("=" * 70)
print("SVM GridSearchCV — BALANCED KDD Cup 99")
print("=" * 70)

print("\n1. Downloading FULL KDD Cup 99 (~18 MB compressed)...")
for name, info in files.items():
    if not os.path.exists(info['csv']):
        if not os.path.exists(info['gz']):
            print(f"   Downloading {name}...")
            urllib.request.urlretrieve(info['url'], info['gz'])
        print(f"   Decompressing {name}...")
        with gzip.open(info['gz'], 'rt') as f:
            content = f.read()
        with open(info['csv'], 'w') as f:
            f.write(content)
        print(f"   Saved: {info['csv']}")
    else:
        print(f"   {name}: already exists")

# ====================================================================
# 2. LOAD FULL DATASET
# ====================================================================
print("\n2. Loading full dataset...")
full_df = pd.read_csv(files['train_full']['csv'], names=col_names, low_memory=False)
test_df = pd.read_csv(files['test']['csv'], names=col_names, low_memory=False)

full_df['label'] = full_df['label'].str.strip()
test_df['label'] = test_df['label'].str.strip()

print(f"   Full train: {len(full_df):,} samples")
print(f"   Test:       {len(test_df):,} samples")

# --- Original distribution ---
print("\n   --- Original Full Dataset Distribution ---")
proto_counts = {}
for proto in ['tcp', 'udp', 'icmp']:
    mask = full_df['protocol_type'] == proto
    n_total = mask.sum()
    n_normal = ((full_df['label'] == 'normal.') & mask).sum()
    n_attack = n_total - n_normal
    proto_counts[proto] = {'total': n_total, 'normal': n_normal, 'attack': n_attack}
    print(f"   {proto.upper():>4}: {n_total:>10,} total  "
          f"(normal: {n_normal:>10,}, attack: {n_attack:>10,})")

# ====================================================================
# 3. BALANCED SAMPLING (same strategy as CAE balanced)
# ====================================================================
#
# UDP:  ALL attacks + 2× normal (bottleneck protocol, take everything)
# TCP/ICMP: up to 10% of full dataset each, 75% attack / 25% normal
# ====================================================================

print("\n3. Creating balanced subset...")

MAX_PER_PROTOCOL = int(len(full_df) * 0.10)

balanced_frames = []

for proto in ['tcp', 'udp', 'icmp']:
    proto_df = full_df[full_df['protocol_type'] == proto]

    proto_normal = proto_df[proto_df['label'] == 'normal.']
    proto_attack = proto_df[proto_df['label'] != 'normal.']

    n_available_normal = len(proto_normal)
    n_available_attack = len(proto_attack)

    if proto == 'udp':
        # UDP: take ALL attacks + 2× as many normals
        n_attack_sample = n_available_attack
        n_normal_sample = min(2 * n_attack_sample, n_available_normal)
    else:
        # TCP/ICMP: 75% attack, 25% normal, up to 10% budget
        max_total = min(MAX_PER_PROTOCOL, len(proto_df))

        n_attack_target = int(max_total * 0.75)
        n_normal_target = max_total - n_attack_target

        n_attack_sample = min(n_attack_target, n_available_attack)
        n_normal_sample = min(n_normal_target, n_available_normal)

        # Fill remaining budget
        total_selected = n_attack_sample + n_normal_sample
        if total_selected < max_total:
            leftover = max_total - total_selected
            extra_attack = min(leftover, n_available_attack - n_attack_sample)
            n_attack_sample += extra_attack
            leftover -= extra_attack
            if leftover > 0:
                extra_normal = min(leftover, n_available_normal - n_normal_sample)
                n_normal_sample += extra_normal

    sampled_attack = proto_attack.sample(n=n_attack_sample, random_state=42)
    sampled_normal = proto_normal.sample(n=n_normal_sample, random_state=42)
    sampled = pd.concat([sampled_attack, sampled_normal])
    balanced_frames.append(sampled)

    print(f"   {proto.upper()} sampled: {len(sampled):>8,}  "
          f"(normal: {n_normal_sample:>7,}, attack: {n_attack_sample:>7,}, "
          f"norm%: {n_normal_sample/len(sampled)*100:.1f}%)")

train_df = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n   BALANCED TRAINING SET: {len(train_df):,} samples "
      f"({len(train_df)/len(full_df)*100:.2f}% of full dataset)")

del full_df, balanced_frames
gc.collect()

# ====================================================================
# 4. VERIFY DISTRIBUTION
# ====================================================================
print("\n4. Final balanced subset distribution:")
print(f"   {'Protocol':<8} {'Total':>10} {'Normal':>10} {'Attack':>10} {'Norm%':>8}")
print("   " + "-" * 50)
for proto in ['tcp', 'udp', 'icmp']:
    mask = train_df['protocol_type'] == proto
    total = mask.sum()
    normal = ((train_df['label'] == 'normal.') & mask).sum()
    attack = total - normal
    pct = normal / total * 100 if total > 0 else 0
    print(f"   {proto.upper():<8} {total:>10,} {normal:>10,} {attack:>10,} {pct:>7.1f}%")
total_all = len(train_df)
normal_all = (train_df['label'] == 'normal.').sum()
print("   " + "-" * 50)
print(f"   {'TOTAL':<8} {total_all:>10,} {normal_all:>10,} {total_all-normal_all:>10,} "
      f"{normal_all/total_all*100:>7.1f}%")

# ====================================================================
# 5. PREPROCESSING
# ====================================================================
print("\n5. Preprocessing (LabelEncoding + MinMax)...")

y_train = (train_df['label'] != 'normal.').astype(int).values
y_test = (test_df['label'] != 'normal.').astype(int).values

feature_cols = [c for c in col_names if c != 'label']

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

X_train = train_df[feature_cols].values.astype(np.float64)
X_test = test_df[feature_cols].values.astype(np.float64)

print(f"   Feature dim: {X_train.shape[1]}")
print(f"   Train: {len(X_train):,}  (normal: {(y_train==0).sum():,}, attack: {(y_train==1).sum():,})")
print(f"   Test:  {len(X_test):,}  (normal: {(y_test==0).sum():,}, attack: {(y_test==1).sum():,})")

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

del train_df, test_df
gc.collect()

# ====================================================================
# 6. GRID SEARCH ON SUBSAMPLE
# ====================================================================
# SVM is O(n²–n³) so grid search on the full balanced set (~500K+)
# would take days. We subsample for hyperparameter search, then
# retrain the best params on the full set.
# ====================================================================

GRID_SEARCH_SIZE = 50000  # Subsample for grid search

print(f"\n6. Grid search on {GRID_SEARCH_SIZE:,} subsample...")

if len(X_train) > GRID_SEARCH_SIZE:
    idx = np.random.RandomState(42).choice(len(X_train), GRID_SEARCH_SIZE, replace=False)
    X_gs = X_train[idx]
    y_gs = y_train[idx]
    print(f"   Subsampled: {len(X_gs):,} (normal: {(y_gs==0).sum():,}, attack: {(y_gs==1).sum():,})")
else:
    X_gs = X_train
    y_gs = y_train
    print(f"   Using full train set (already <= {GRID_SEARCH_SIZE:,})")

# Parameter grid
param_grid = [
    # RBF kernel
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    },
    # Linear kernel
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    },
    # Polynomial kernel
    {
        'kernel': ['poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3]
    }
]

svm = SVC(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    svm,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("\n   Starting GridSearchCV...")
t0 = time.time()
grid_search.fit(X_gs, y_gs)
gs_time = time.time() - t0
print(f"   Grid search completed in {gs_time:.1f}s ({gs_time/60:.1f} min)")

print(f"\n   Best Parameters: {grid_search.best_params_}")
print(f"   Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

# Top 5
results_df = pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score')
print("\n   Top 5 Parameter Combinations:")
for _, row in results_df.head(5).iterrows():
    print(f"     #{int(row['rank_test_score'])}. {row['params']} -> {row['mean_test_score']*100:.2f}%")

# ====================================================================
# 7. RETRAIN BEST MODEL ON FULL BALANCED SET
# ====================================================================
print(f"\n7. Retraining best model on full balanced set ({len(X_train):,} samples)...")

best_params = grid_search.best_params_
print(f"   Params: {best_params}")

# For very large datasets, cap training size for SVM feasibility
MAX_RETRAIN = 200000
if len(X_train) > MAX_RETRAIN:
    print(f"   Capping retrain to {MAX_RETRAIN:,} samples (SVM memory constraint)")
    idx_rt = np.random.RandomState(42).choice(len(X_train), MAX_RETRAIN, replace=False)
    X_retrain = X_train[idx_rt]
    y_retrain = y_train[idx_rt]
else:
    X_retrain = X_train
    y_retrain = y_train

best_svm = SVC(**best_params, random_state=42, probability=True)

t0 = time.time()
best_svm.fit(X_retrain, y_retrain)
train_time = time.time() - t0
print(f"   Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")

# ====================================================================
# 8. EVALUATE ON TEST SET
# ====================================================================
print(f"\n{'='*70}")
print("TEST SET EVALUATION — SVM on Balanced KDD Cup 99")
print(f"{'='*70}")

y_pred = best_svm.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
fpr_val = fp / (fp + tn)
tpr_val = tp / (tp + fn)

print(f"\n   Accuracy:  {accuracy * 100:.2f}%")
print(f"   FPR:       {fpr_val * 100:.2f}%")
print(f"   TPR:       {tpr_val * 100:.2f}%")
print(f"\n   Confusion Matrix:")
print(f"     TN={tn:,}, FP={fp:,}")
print(f"     FN={fn:,}, TP={tp:,}")

# AUC via decision_function (more reliable than probability for SVM)
try:
    y_scores = best_svm.decision_function(X_test)
    auc = roc_auc_score(y_test, y_scores)
    print(f"   AUC:       {auc:.4f}")
except Exception:
    try:
        y_proba = best_svm.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"   AUC:       {auc:.4f}")
    except Exception:
        auc = float('nan')
        print(f"   AUC:       N/A")

print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])}")

# ====================================================================
# 9. PAPER COMPARISON
# ====================================================================
# Paper Table II (SVM): Accuracy=69.83%, FPR=17.35%
# Paper Table I (SVM per-protocol AUC): TCP=0.6753, UDP=0.9890, ICMP=0.9350, Overall=0.8574
paper_acc = 69.83
paper_fpr = 17.35
paper_auc_overall = 0.8574

print(f"\n{'='*70}")
print("COMPARISON WITH PAPER")
print(f"{'='*70}")
print(f"   {'Metric':<20} {'Ours':>10} {'Paper':>10} {'Diff':>10}")
print("   " + "-" * 52)
print(f"   {'Accuracy (%)':<20} {accuracy*100:>10.2f} {paper_acc:>10.2f} {accuracy*100 - paper_acc:>+10.2f}")
print(f"   {'FPR (%)':<20} {fpr_val*100:>10.2f} {paper_fpr:>10.2f} {fpr_val*100 - paper_fpr:>+10.2f}")
if not np.isnan(auc):
    print(f"   {'AUC':<20} {auc:>10.4f} {paper_auc_overall:>10.4f} {auc - paper_auc_overall:>+10.4f}")

# ====================================================================
# 10. PLOT
# ====================================================================
print("\n10. Generating results plot...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('SVM Results vs Paper (Chen et al., WTS 2018)\n'
             'Balanced KDD Cup 99', fontsize=14, fontweight='bold')

c_ours = '#2563EB'
c_paper = '#DC2626'

# Plot 1: Accuracy & FPR comparison
ax1 = axes[0]
metrics = ['Accuracy', 'FPR']
ours_vals = [accuracy * 100, fpr_val * 100]
paper_vals = [paper_acc, paper_fpr]
x = np.arange(len(metrics))
w = 0.32

b1 = ax1.bar(x - w/2, ours_vals, w, label='Ours (balanced)', color=c_ours, edgecolor='white')
b2 = ax1.bar(x + w/2, paper_vals, w, label='Paper', color=c_paper, edgecolor='white')
ax1.set_ylabel('Percentage (%)', fontsize=11)
ax1.set_title('Accuracy & FPR', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: AUC comparison
ax2 = axes[1]
if not np.isnan(auc):
    auc_labels = ['Overall AUC']
    ours_aucs = [auc]
    paper_aucs = [paper_auc_overall]
    x2 = np.arange(len(auc_labels))

    b3 = ax2.bar(x2 - w/2, ours_aucs, w, label='Ours', color=c_ours, edgecolor='white')
    b4 = ax2.bar(x2 + w/2, paper_aucs, w, label='Paper', color=c_paper, edgecolor='white')
    ax2.set_ylabel('AUC Score', fontsize=11)
    ax2.set_title('AUC Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(auc_labels, fontsize=11)
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for bar in b3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in b4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'AUC not available', ha='center', va='center',
             fontsize=14, transform=ax2.transAxes)

plt.tight_layout()
plt.savefig('./svm_kdd99_balanced_results.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("   Saved: svm_kdd99_balanced_results.png")

# ====================================================================
# 11. SAVE GRID SEARCH RESULTS
# ====================================================================
print("\n11. Grid search summary:")
print(f"    Best params: {best_params}")
print(f"    Best CV accuracy: {grid_search.best_score_*100:.2f}%")
print(f"    Test accuracy: {accuracy*100:.2f}%")
print(f"    Grid search time: {gs_time:.1f}s")
print(f"    Retrain time: {train_time:.1f}s")

print("\n[DONE] SVM on balanced KDD Cup 99 complete.")