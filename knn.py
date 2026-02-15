import pandas as pd
import numpy as np
import os
import gzip
import urllib.request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)

# ====================================================================
# 1. DOWNLOAD KDD CUP 99 DATA
# ====================================================================
DATA_DIR = './kddcup99_data'
os.makedirs(DATA_DIR, exist_ok=True)

files = {
    'train': {
        'url': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz',
        'gz': os.path.join(DATA_DIR, 'kddcup.data_10_percent.gz'),
        'csv': os.path.join(DATA_DIR, 'kddcup_train.csv'),
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

print("=" * 60)
print("k-NEAREST NEIGHBOR (k-NN) - Network Anomaly Detection")
print("Training on KDD Cup 99 (as in original paper references)")
print("=" * 60)

# Download
print("\n1. Downloading KDD Cup 99 data...")
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
# 2. LOAD DATA
# ====================================================================
print("\n2. Loading data...")
train_df = pd.read_csv(files['train']['csv'], names=col_names, low_memory=False)
test_df = pd.read_csv(files['test']['csv'], names=col_names, low_memory=False)

train_df['label'] = train_df['label'].str.strip()
test_df['label'] = test_df['label'].str.strip()

print(f"   KDD99 Train (10%): {len(train_df)} samples")
print(f"   KDD99 Test (corrected): {len(test_df)} samples")

# ====================================================================
# 3. PREPROCESSING
# ====================================================================
print("\n3. Processing Labels (Binary: Normal=0, Attack=1)...")
y_train = (train_df['label'] != 'normal.').astype(int).values
y_test = (test_df['label'] != 'normal.').astype(int).values

print(f"   Train - Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
print(f"   Test  - Normal: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")

print("\n4. Encoding Categorical Features (LabelEncoder)...")
categorical_cols = ['protocol_type', 'service', 'flag']
feature_cols = [c for c in col_names if c != 'label']

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

print("   Applying MinMax Scaling to [0,1]...")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"   Feature dimension: {X_train.shape[1]}")

# ====================================================================
# 4. TEST DIFFERENT k VALUES
# ====================================================================
print("\n5. Testing different k values...")
k_values = [ 15]

print(f"\n{'k':<5} {'Accuracy':<12} {'FPR':<10} {'TPR':<10}")
print("-" * 40)

best_k = 1
best_acc = 0

for k in k_values:
    knn_tmp = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn_tmp.fit(X_train, y_train)
    y_pred_tmp = knn_tmp.predict(X_test)
    
    acc_tmp = accuracy_score(y_test, y_pred_tmp)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_tmp).ravel()
    fpr_tmp = fp_t / (fp_t + tn_t)
    tpr_tmp = tp_t / (tp_t + fn_t)
    
    print(f"{k:<5} {acc_tmp*100:.2f}%      {fpr_tmp*100:.2f}%     {tpr_tmp*100:.2f}%")
    
    if acc_tmp > best_acc:
        best_acc = acc_tmp
        best_k = k

print(f"\nBest k = {best_k} with accuracy = {best_acc*100:.2f}%")

# ====================================================================
# 5. FINAL MODEL
# ====================================================================
print(f"\n6. Training final k-NN (k={best_k})...")
knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn.fit(X_train, y_train)

print("7. Predicting on test set...")
y_pred = knn.predict(X_test)

# ====================================================================
# 6. RESULTS ON KDD CUP 99
# ====================================================================
acc = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)
tpr = tp / (tp + fn)

print("\n" + "=" * 60)
print("RESULTS ON KDD CUP 99 (Original dataset)")
print("=" * 60)
print(f"Detection Accuracy:      {acc * 100:.2f}%   (Paper: 88.91%)")
print(f"False Positive Rate:     {fpr * 100:.2f}%   (Paper: 38.02%)")
print(f"True Positive Rate:      {tpr * 100:.2f}%")
print(f"Best k:                  {best_k}")
print("-" * 60)
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print("-" * 60)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# AUC
print("\nAUC SCORES BY PROTOCOL:")
print("-" * 40)

# Save original protocols before encoding
test_df_orig = pd.read_csv(files['test']['csv'], names=col_names, low_memory=False)
test_df_orig['label'] = test_df_orig['label'].str.strip()

y_pred_proba = knn.predict_proba(X_test)[:, 1]
protocols = ['tcp', 'udp', 'icmp']

print(f"{'Protocol':<10} {'AUC Score':<12} {'Sample Size':<12}")
print("-" * 40)

for proto in protocols:
    mask = (test_df_orig['protocol_type'] == proto).values
    if mask.sum() > 0:
        y_sub = y_test[mask]
        proba_sub = y_pred_proba[mask]
        if len(np.unique(y_sub)) > 1:
            auc = roc_auc_score(y_sub, proba_sub)
            print(f"{proto.upper():<10} {auc:.4f}       {mask.sum():<12}")
        else:
            print(f"{proto.upper():<10} {'N/A':<12} {mask.sum():<12}")

auc_all = roc_auc_score(y_test, y_pred_proba)
print("-" * 40)
print(f"{'OVERALL':<10} {auc_all:.4f}       {len(y_test):<12}")

# ====================================================================
# 7. BONUS: NSL-KDD EVALUATION
# ====================================================================
nsl_train_path = './KDDTrain+.txt'
nsl_test_path = './KDDTest+.txt'
nsl_col_names = col_names + ["difficulty"]

print("\n" + "=" * 60)
print("BONUS: NSL-KDD Evaluation")
print("=" * 60)

if os.path.exists(nsl_train_path) and os.path.exists(nsl_test_path):
    print("Loading NSL-KDD...")
    nsl_train = pd.read_csv(nsl_train_path, names=nsl_col_names)
    nsl_test = pd.read_csv(nsl_test_path, names=nsl_col_names)
    
    nsl_train.drop('difficulty', axis=1, inplace=True)
    nsl_test.drop('difficulty', axis=1, inplace=True)
    
    # Save original protocol
    nsl_test_proto_orig = nsl_test['protocol_type'].values.copy()
    
    y_nsl_train = (nsl_train['label'] != 'normal').astype(int).values
    y_nsl_test = (nsl_test['label'] != 'normal').astype(int).values
    
    for col in categorical_cols:
        le2 = LabelEncoder()
        le2.fit(pd.concat([nsl_train[col], nsl_test[col]]))
        nsl_train[col] = le2.transform(nsl_train[col])
        nsl_test[col] = le2.transform(nsl_test[col])
    
    feature_cols_nsl = [c for c in col_names if c != 'label']
    X_nsl_train = nsl_train[feature_cols_nsl].values
    X_nsl_test = nsl_test[feature_cols_nsl].values
    
    scaler_nsl = MinMaxScaler()
    X_nsl_train = scaler_nsl.fit_transform(X_nsl_train)
    X_nsl_test = scaler_nsl.transform(X_nsl_test)
    
    # Test k values on NSL-KDD
    print("\nTesting k values on NSL-KDD...")
    print(f"\n{'k':<5} {'Accuracy':<12} {'FPR':<10} {'TPR':<10}")
    print("-" * 40)
    
    best_k_nsl = 1
    best_acc_nsl = 0
    
    for k in k_values:
        knn_n = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_n.fit(X_nsl_train, y_nsl_train)
        y_p = knn_n.predict(X_nsl_test)
        
        a = accuracy_score(y_nsl_test, y_p)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_nsl_test, y_p).ravel()
        f = fp2 / (fp2 + tn2)
        t = tp2 / (tp2 + fn2)
        
        print(f"{k:<5} {a*100:.2f}%      {f*100:.2f}%     {t*100:.2f}%")
        
        if a > best_acc_nsl:
            best_acc_nsl = a
            best_k_nsl = k
    
    print(f"\nBest k = {best_k_nsl} with accuracy = {best_acc_nsl*100:.2f}%")
    
    # Final NSL-KDD model
    knn_nsl = KNeighborsClassifier(n_neighbors=best_k_nsl, n_jobs=-1)
    knn_nsl.fit(X_nsl_train, y_nsl_train)
    y_nsl_pred = knn_nsl.predict(X_nsl_test)
    
    acc_nsl = accuracy_score(y_nsl_test, y_nsl_pred)
    tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_nsl_test, y_nsl_pred).ravel()
    fpr_nsl = fp_n / (fp_n + tn_n)
    tpr_nsl = tp_n / (tp_n + fn_n)
    
    print(f"\n{'='*60}")
    print(f"NSL-KDD RESULTS")
    print(f"{'='*60}")
    print(f"Detection Accuracy:      {acc_nsl * 100:.2f}%")
    print(f"False Positive Rate:     {fpr_nsl * 100:.2f}%")
    print(f"True Positive Rate:      {tpr_nsl * 100:.2f}%")
    print(f"Best k:                  {best_k_nsl}")
    print("-" * 60)
    print(f"Confusion Matrix: TN={tn_n}, FP={fp_n}, FN={fn_n}, TP={tp_n}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_nsl_test, y_nsl_pred, target_names=['Normal', 'Attack']))
    
    # NSL-KDD AUC by protocol
    y_nsl_proba = knn_nsl.predict_proba(X_nsl_test)[:, 1]
    
    print("AUC SCORES BY PROTOCOL (NSL-KDD):")
    print(f"{'Protocol':<10} {'AUC Score':<12} {'Sample Size':<12}")
    print("-" * 40)
    for proto in protocols:
        mask = (nsl_test_proto_orig == proto)
        if mask.sum() > 0:
            y_sub = y_nsl_test[mask]
            proba_sub = y_nsl_proba[mask]
            if len(np.unique(y_sub)) > 1:
                auc = roc_auc_score(y_sub, proba_sub)
                print(f"{proto.upper():<10} {auc:.4f}       {mask.sum():<12}")
    auc_nsl = roc_auc_score(y_nsl_test, y_nsl_proba)
    print("-" * 40)
    print(f"{'OVERALL':<10} {auc_nsl:.4f}       {len(y_nsl_test):<12}")
else:
    print("NSL-KDD files not found, skipping.")

print("\n[DONE] k-NN evaluation complete.")