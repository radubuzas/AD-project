import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, classification_report, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import gzip
import urllib.request
import gc

np.random.seed(42)
tf.random.set_seed(42)

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

print("=" * 70)
print("CAE PER-PROTOCOL — BALANCED KDD Cup 99 (~10%, equal protocols)")
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
# 2. LOAD FULL DATASET & CREATE BALANCED SUBSET
# ====================================================================
print("\n2. Loading full dataset...")
full_df = pd.read_csv(files['train_full']['csv'], names=col_names, low_memory=False)
test_df = pd.read_csv(files['test']['csv'], names=col_names, low_memory=False)

full_df['label'] = full_df['label'].str.strip()
test_df['label'] = test_df['label'].str.strip()

print(f"   Full train: {len(full_df):,} samples")
print(f"   Test:       {len(test_df):,} samples")

# --- Show original distribution ---
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
# 3. BALANCED SAMPLING STRATEGY
# ====================================================================
#
# Strategy:
#   1. Find the smallest protocol (UDP, typically ~20K samples)
#   2. Take ALL of it (can't subsample what's already small)
#   3. Sample TCP and ICMP down to the SAME size as UDP
#   4. Within each protocol, preserve the normal/attack ratio (stratified)
#   5. Total = 3 × UDP_size
#
# Then fill remaining budget (to reach ~10%) with extra TCP samples,
# since TCP is the dominant protocol and benefits from more data.
# ====================================================================

print("\n3. Creating balanced subset...")

# Find the bottleneck protocol
smallest_proto = min(proto_counts, key=lambda p: proto_counts[p]['total'])
smallest_count = proto_counts[smallest_proto]['total']

print(f"   Bottleneck protocol: {smallest_proto.upper()} ({smallest_count:,} samples)")

TARGET_TOTAL = int(len(full_df) * 0.10)
per_proto_base = smallest_count
remaining = TARGET_TOTAL - (3 * per_proto_base)

# Distribute: equal base per protocol + extra to TCP if budget allows
proto_targets = {}
for proto in ['tcp', 'udp', 'icmp']:
    proto_targets[proto] = per_proto_base
if remaining > 0:
    proto_targets['tcp'] += remaining

print(f"   Target total: ~{TARGET_TOTAL:,} (10% of {len(full_df):,})")
print(f"   Sampling plan:")
for proto in ['tcp', 'udp', 'icmp']:
    avail = proto_counts[proto]['total']
    target = min(proto_targets[proto], avail)
    proto_targets[proto] = target
    print(f"     {proto.upper()}: {target:>10,} / {avail:>10,} available")

# --- Stratified sampling ---
MAX_PER_PROTOCOL = int(len(full_df) * 0.10)

balanced_frames = []

for proto in ['tcp', 'udp', 'icmp']:
    proto_df = full_df[full_df['protocol_type'] == proto]

    proto_normal = proto_df[proto_df['label'] == 'normal.']
    proto_attack = proto_df[proto_df['label'] != 'normal.']

    n_available_normal = len(proto_normal)
    n_available_attack = len(proto_attack)

    # -------------------------------
    # SPECIAL RULE FOR UDP
    # -------------------------------
    if proto == 'udp':
        # Load ALL attack samples
        n_attack_sample = n_available_attack

        # Take twice as many normal samples (if available)
        n_normal_sample = min(2 * n_attack_sample, n_available_normal)

    # -------------------------------
    # ORIGINAL LOGIC FOR OTHERS
    # -------------------------------
    else:
        max_total = min(MAX_PER_PROTOCOL, len(proto_df))

        n_attack_target = int(max_total * 0.75)
        n_normal_target = max_total - n_attack_target

        n_attack_sample = min(n_attack_target, n_available_attack)
        n_normal_sample = min(n_normal_target, n_available_normal)

        total_selected = n_attack_sample + n_normal_sample

        if total_selected < max_total:
            remaining = max_total - total_selected

            extra_attack = min(remaining, n_available_attack - n_attack_sample)
            n_attack_sample += extra_attack
            remaining -= extra_attack

            if remaining > 0:
                extra_normal = min(remaining, n_available_normal - n_normal_sample)
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

train_protocols = train_df['protocol_type'].values.copy()
test_protocols = test_df['protocol_type'].values.copy()

y_train_all = (train_df['label'] != 'normal.').astype(int)
y_test = (test_df['label'] != 'normal.').astype(int)

categorical_cols = ['protocol_type', 'service', 'flag']
feature_cols = [c for c in col_names if c != 'label']

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

X_train_all = train_df[feature_cols].values.astype(np.float32)
X_test_all = test_df[feature_cols].values.astype(np.float32)
input_dim = X_train_all.shape[1]

print(f"   Feature dim: {input_dim}")

del train_df, test_df
gc.collect()

# ====================================================================
# 6. HELPERS
# ====================================================================

def make_tam_dataset(X_1d, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(X_1d)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X_1d), 10000), seed=42)
    ds = ds.batch(batch_size)

    def tam_transform(batch):
        tam = tf.einsum('bi,bj->bij', batch, batch)
        tam = tf.expand_dims(tam, -1)
        return tam, tam

    ds = ds.map(tam_transform, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_cae(dim, name='cae'):
    """Paper Fig. 3: single Conv2D encoder + single Conv2DTranspose decoder."""
    inp = keras.Input(shape=(dim, dim, 1))
    x = layers.Conv2D(16, (5, 5), strides=1, padding='same', activation='sigmoid')(inp)
    x = layers.Conv2DTranspose(1, (5, 5), strides=1, padding='same', activation='sigmoid')(x)
    model = keras.Model(inp, x, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def compute_recon_error(model, X_1d, batch_size=128):
    errors = []
    for i in range(0, len(X_1d), batch_size):
        batch = X_1d[i:i + batch_size]
        tam = np.einsum('bi,bj->bij', batch, batch)[..., np.newaxis]
        recon = model.predict(tam, verbose=0)
        mse = np.mean(np.square(tam - recon).reshape(len(batch), -1), axis=1)
        errors.append(mse)
    return np.concatenate(errors)


# ====================================================================
# 7. PER-PROTOCOL TRAINING & EVALUATION
# ====================================================================
protocols = ['tcp', 'udp', 'icmp']
paper_auc = {'tcp': 0.9902, 'udp': 0.9564, 'icmp': 0.9869}
paper_acc = 96.87
paper_fpr = 3.44

BATCH_SIZE = 64
EPOCHS = 8

results = {}

for proto in protocols:
    print(f"\n{'='*70}")
    print(f"  TRAINING CAE FOR PROTOCOL: {proto.upper()}")
    print(f"{'='*70}")

    train_mask = (train_protocols == proto)
    test_mask = (test_protocols == proto)

    X_train_proto = X_train_all[train_mask]
    y_train_proto = y_train_all.values[train_mask]
    X_test_proto = X_test_all[test_mask]
    y_test_proto = y_test.values[test_mask]

    normal_mask = (y_train_proto == 0)
    X_normal = X_train_proto[normal_mask]

    print(f"   Train total: {len(X_train_proto):,} "
          f"(normal: {normal_mask.sum():,}, attack: {(~normal_mask).sum():,})")
    print(f"   Test total:  {len(X_test_proto):,} "
          f"(normal: {(y_test_proto==0).sum():,}, attack: {(y_test_proto==1).sum():,})")

    # Per-protocol MinMax scaling (fit on normal only)
    scaler_p = MinMaxScaler()
    scaler_p.fit(X_normal)
    X_normal_scaled = np.clip(scaler_p.transform(X_normal), 0, 1).astype(np.float32)
    X_test_scaled = np.clip(scaler_p.transform(X_test_proto), 0, 1).astype(np.float32)

    # Validation split (last 10%)
    val_size = max(int(len(X_normal_scaled) * 0.1), 1)
    X_tr = X_normal_scaled[:-val_size]
    X_vl = X_normal_scaled[-val_size:]

    train_ds = make_tam_dataset(X_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_tam_dataset(X_vl, batch_size=BATCH_SIZE, shuffle=False)

    model = build_cae(input_dim, name=f'cae_{proto}')
    print(f"   Model params: {model.count_params():,}")

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=[early_stop], verbose=1
    )

    recon_err = compute_recon_error(model, X_test_scaled, batch_size=128)

    if len(np.unique(y_test_proto)) > 1:
        auc = roc_auc_score(y_test_proto, recon_err)
        fpr_c, tpr_c, thr_c = roc_curve(y_test_proto, recon_err)
        best_j = np.argmax(tpr_c - fpr_c)
        threshold = thr_c[best_j]
        y_pred = (recon_err > threshold).astype(int)

        acc = accuracy_score(y_test_proto, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test_proto, y_pred).ravel()
        fpr_val = fp / (fp + tn)
        tpr_val = tp / (tp + fn)
    else:
        auc = acc = fpr_val = tpr_val = float('nan')
        tn = fp = fn = tp = 0

    results[proto] = {
        'auc': auc, 'acc': acc, 'fpr': fpr_val, 'tpr': tpr_val,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'test_size': len(X_test_proto),
        'train_normal': normal_mask.sum(),
        'epochs_trained': len(history.history['loss']),
        'final_val_loss': history.history['val_loss'][-1],
    }

    print(f"\n   --- {proto.upper()} Results ---")
    print(f"   AUC:      {auc:.4f}  (Paper: {paper_auc[proto]:.4f})")
    print(f"   Accuracy: {acc*100:.2f}%")
    print(f"   FPR:      {fpr_val*100:.2f}%")
    print(f"   TPR:      {tpr_val*100:.2f}%")
    print(f"   CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"   Epochs:   {results[proto]['epochs_trained']}")

    del model, train_ds, val_ds
    keras.backend.clear_session()
    gc.collect()

# ====================================================================
# 8. OVERALL CAE (all protocols combined)
# ====================================================================
print(f"\n{'='*70}")
print("TRAINING OVERALL CAE (all protocols combined)")
print(f"{'='*70}")

train_normal_mask_all = (y_train_all == 0).values
X_normal_all = X_train_all[train_normal_mask_all]

scaler_all = MinMaxScaler()
scaler_all.fit(X_normal_all)
X_normal_all_s = np.clip(scaler_all.transform(X_normal_all), 0, 1).astype(np.float32)
X_test_all_s = np.clip(scaler_all.transform(X_test_all), 0, 1).astype(np.float32)

val_sz = int(len(X_normal_all_s) * 0.1)
train_ds_all = make_tam_dataset(X_normal_all_s[:-val_sz], batch_size=BATCH_SIZE, shuffle=True)
val_ds_all = make_tam_dataset(X_normal_all_s[-val_sz:], batch_size=BATCH_SIZE, shuffle=False)

model_all = build_cae(input_dim, name='cae_all')
model_all.fit(
    train_ds_all, validation_data=val_ds_all,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

recon_all = compute_recon_error(model_all, X_test_all_s, batch_size=128)
overall_auc = roc_auc_score(y_test, recon_all)

fpr_c, tpr_c, thr_c = roc_curve(y_test, recon_all)
best_j = np.argmax(tpr_c - fpr_c)
y_pred_all = (recon_all > thr_c[best_j]).astype(int)
overall_acc = accuracy_score(y_test, y_pred_all)
tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test, y_pred_all).ravel()
overall_fpr = fp_a / (fp_a + tn_a)
overall_tpr = tp_a / (tp_a + fn_a)

results['overall'] = {
    'auc': overall_auc, 'acc': overall_acc,
    'fpr': overall_fpr, 'tpr': overall_tpr,
    'tn': tn_a, 'fp': fp_a, 'fn': fn_a, 'tp': tp_a,
    'test_size': len(y_test),
}

paper_auc['overall'] = 0.9884

del model_all, train_ds_all, val_ds_all
keras.backend.clear_session()
gc.collect()

# ====================================================================
# 9. SUMMARY TABLE
# ====================================================================
print(f"\n{'='*70}")
print("COMPLETE RESULTS — CAE on BALANCED KDD Cup 99 Subset")
print(f"{'='*70}")

print(f"\n{'Protocol':<10} {'Our AUC':<10} {'Paper AUC':<11} {'Diff':<9} "
      f"{'Acc%':<9} {'FPR%':<9} {'TPR%':<9} {'Train N':<10} {'Test N':<8}")
print("-" * 90)

for key in ['tcp', 'udp', 'icmp', 'overall']:
    r = results[key]
    p_auc = paper_auc[key]
    diff = r['auc'] - p_auc
    sign = '+' if diff >= 0 else ''
    train_n = r.get('train_normal', '—')
    print(f"{key.upper():<10} {r['auc']:.4f}     {p_auc:.4f}      {sign}{diff:.4f}   "
          f"{r['acc']*100:.2f}    {r['fpr']*100:.2f}    {r['tpr']*100:.2f}    "
          f"{str(train_n):<10} {r['test_size']}")

print("-" * 90)
print(f"\nPaper Table II: Accuracy=96.87%, FPR=3.44%")
print(f"Our Overall:   Accuracy={overall_acc*100:.2f}%, FPR={overall_fpr*100:.2f}%")

for proto in protocols:
    r = results[proto]
    print(f"\n--- {proto.upper()} Confusion Matrix ---")
    print(f"TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}")

print(f"\n--- OVERALL Classification Report ---")
print(classification_report(y_test, y_pred_all, target_names=['Normal', 'Attack']))

# ====================================================================
# 10. PLOTS
# ====================================================================
print("\n10. Generating comparison plots...")

labels = ['TCP', 'UDP', 'ICMP', 'Overall']
keys = ['tcp', 'udp', 'icmp', 'overall']
our_aucs = [results[k]['auc'] for k in keys]
pap_aucs = [paper_auc[k] for k in keys]
our_accs = [results[k]['acc'] * 100 for k in keys]
our_fprs = [results[k]['fpr'] * 100 for k in keys]
our_tprs = [results[k]['tpr'] * 100 for k in keys]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CAE Per-Protocol Results vs Paper (Chen et al., WTS 2018)\n'
             'Balanced KDD Cup 99 (all UDP + matched TCP/ICMP)',
             fontsize=14, fontweight='bold', y=0.98)

bar_width = 0.32
x = np.arange(len(labels))
c_ours = '#2563EB'
c_paper = '#DC2626'
c_acc = '#059669'
c_fpr = '#D97706'

# Plot 1: AUC Comparison
ax1 = axes[0, 0]
b1 = ax1.bar(x - bar_width/2, our_aucs, bar_width, label='Ours (balanced)', color=c_ours, edgecolor='white', linewidth=0.5)
b2 = ax1.bar(x + bar_width/2, pap_aucs, bar_width, label='Paper', color=c_paper, edgecolor='white', linewidth=0.5)
ax1.set_ylabel('AUC Score', fontsize=11)
ax1.set_title('AUC: Ours vs Paper', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylim(0.85, 1.008)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

# Plot 2: AUC Difference
ax2 = axes[0, 1]
diffs = [o - p for o, p in zip(our_aucs, pap_aucs)]
colors_diff = [c_ours if d >= 0 else c_paper for d in diffs]
b3 = ax2.bar(x, diffs, bar_width * 1.5, color=colors_diff, edgecolor='white', linewidth=0.5)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_ylabel('AUC Difference (Ours − Paper)', fontsize=11)
ax2.set_title('AUC Difference', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for bar, d in zip(b3, diffs):
    offset = 0.001 if d >= 0 else -0.003
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
             f'{d:+.4f}', ha='center', va='bottom' if d >= 0 else 'top', fontsize=9, fontweight='bold')

# Plot 3: Accuracy & TPR
ax3 = axes[1, 0]
b4 = ax3.bar(x - bar_width/2, our_accs, bar_width, label='Accuracy', color=c_acc, edgecolor='white', linewidth=0.5)
b5 = ax3.bar(x + bar_width/2, our_tprs, bar_width, label='TPR (Recall)', color=c_ours, edgecolor='white', linewidth=0.5)
ax3.axhline(y=paper_acc, color=c_paper, linewidth=1.2, linestyle='--', label=f'Paper Acc ({paper_acc}%)')
ax3.set_ylabel('Percentage (%)', fontsize=11)
ax3.set_title('Per-Protocol Accuracy & TPR', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=10)
ax3.set_ylim(50, 105)
ax3.legend(fontsize=9, loc='lower left')
ax3.grid(axis='y', alpha=0.3)
for bar in b4:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in b5:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

# Plot 4: FPR
ax4 = axes[1, 1]
b6 = ax4.bar(x, our_fprs, bar_width * 1.5, color=c_fpr, edgecolor='white', linewidth=0.5)
ax4.axhline(y=paper_fpr, color=c_paper, linewidth=1.2, linestyle='--', label=f'Paper FPR ({paper_fpr}%)')
ax4.set_ylabel('False Positive Rate (%)', fontsize=11)
ax4.set_title('Per-Protocol FPR', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)
for bar in b6:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('./cae_kdd99_balanced_results.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("   Saved: cae_kdd99_balanced_results.png")

# Operating Points
fig2, ax = plt.subplots(figsize=(8, 5))
proto_colors = {'tcp': '#2563EB', 'udp': '#059669', 'icmp': '#D97706', 'overall': '#7C3AED'}

for key in keys:
    r = results[key]
    ax.scatter(r['fpr']*100, r['tpr']*100, s=120, color=proto_colors[key],
               zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(f"{key.upper()}\nAUC={r['auc']:.4f}",
                (r['fpr']*100, r['tpr']*100),
                textcoords="offset points", xytext=(12, -5), fontsize=9,
                fontweight='bold', color=proto_colors[key])

ax.scatter([paper_fpr], [100 - paper_fpr], s=120, color=c_paper, marker='D',
           zorder=5, edgecolors='white', linewidth=1.5)
ax.annotate(f"Paper\nAcc={paper_acc}%", (paper_fpr, 100 - paper_fpr),
            textcoords="offset points", xytext=(12, -5), fontsize=9,
            fontweight='bold', color=c_paper)

ax.set_xlabel('False Positive Rate (%)', fontsize=12)
ax.set_ylabel('True Positive Rate (%)', fontsize=12)
ax.set_title('Operating Points: Per-Protocol CAE vs Paper\nBalanced KDD Cup 99',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xlim(-2, max(our_fprs + [paper_fpr]) + 15)
ax.set_ylim(min(our_tprs) - 5, 102)

plt.tight_layout()
plt.savefig('./cae_kdd99_balanced_operating_points.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("   Saved: cae_kdd99_balanced_operating_points.png")

print("\n[DONE] Per-protocol CAE on balanced KDD Cup 99 complete.")