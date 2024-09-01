# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.model_selection as skl
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import os
from random import sample

def draw(y_test, y_pred_rf, save_path, data_name):
    # AUROC
    fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(y_test, y_pred_rf)
    roc_auc = metrics.auc(fpr_rf_lm, tpr_rf_lm)
    plt.figure()
    plt.plot(fpr_rf_lm, tpr_rf_lm, color='darkorange', lw=2, label=f'{data_name} ROC curve (area = {roc_auc:.5f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{data_name} AUROC Curve')
    plt.legend(loc="lower right")
    print(f'{data_name} AUROC: ', roc_auc)

    # Save AUROC Curve
    plt.savefig(os.path.join(save_path, f'{data_name}_auroc_curve.png'))
    plt.close()

    # AUPRC
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_rf)
    auprc = metrics.average_precision_score(y_test, y_pred_rf)
    plt.figure()
    plt.step(recall, precision, color='b', where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{data_name} Precision-Recall Curve (AUPRC = {auprc:.5f})')
    print(f'{data_name} AP: ', auprc)
    print(f'{data_name} Area under the precision recall curve: ', metrics.auc(recall, precision))

    # Save Precision-Recall Curve
    plt.savefig(os.path.join(save_path, f'{data_name}_precision_recall_curve.png'))
    plt.close()

    # Calculate F1 Score for different thresholds
    thresholds = np.arange(0, 1, 0.01)
    f1_scores = []

    for threshold in thresholds:
        f1_score = metrics.f1_score(y_test, (y_pred_rf > threshold).astype(int))
        f1_scores.append(f1_score)
    index = int(0.5 / 0.01)  # 计算阈值0.5所在的索引位置
    f1_score_at_05_threshold = f1_scores[index]
    print("F1 Score at threshold 0.5:", f1_score_at_05_threshold)

    # Plot F1 Score vs. Threshold
    plt.figure()
    plt.plot(thresholds, f1_scores, label=f'{data_name} F1 Score vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'{data_name} F1 Score vs. Threshold')
    plt.legend(loc="lower right")

    # Find the threshold that maximizes F1 Score
    max_f1_score = max(f1_scores)
    optimal_threshold = thresholds[f1_scores.index(max_f1_score)]
    print(f'{data_name} Max F1 Score: {max_f1_score:.5f} at Threshold: {optimal_threshold:.5f}')

    # Save F1 Score vs. Threshold Curve
    plt.savefig(os.path.join(save_path, f'{data_name}_f1_score_vs_threshold.png'))
    plt.close()

    plt.show()

def train_and_evaluate(data_file, ae_file, dmde_file):
    save_path = 'AUPP'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    DATASETDIR = os.path.expanduser('../data')
    df = pd.read_csv(os.path.join(DATASETDIR, data_file))
    df_ae = pd.read_csv(os.path.join(DATASETDIR, ae_file))
    df_dmde = pd.read_csv(os.path.join(DATASETDIR, dmde_file))

    # Rename columns for consistency
    df.columns = [str(i) for i in range(df.shape[1])]
    df_ae.columns = [str(i) for i in range(df_ae.shape[1])]
    df_dmde.columns = [str(i) for i in range(df_dmde.shape[1])]

    # Rename target column
    df.rename(columns={'34': 'target', '35': 'target'}, inplace=True)
    df_ae.rename(columns={'34': 'target'}, inplace=True)
    df_dmde.rename(columns={'34': 'target'}, inplace=True)

    df['target'] = df['target'].astype(int)
    df_ae['target'] = df_ae['target'].astype(int)
    df_dmde['target'] = df_dmde['target'].astype(int)

    # Train/test split
    train, test = skl.train_test_split(df, test_size=0.2, stratify=df['target'])
    train_ae, test_ae = skl.train_test_split(df_ae, test_size=0.2, stratify=df_ae['target'])
    train_dmde, test_dmde = skl.train_test_split(df_dmde, test_size=0.2, stratify=df_dmde['target'])

    # Make equal number of positive and negative samples in training
    def balance_classes(train_data, label_col='target'):
        pos_indices = train_data[train_data[label_col] == 1].index.tolist()
        neg_indices = train_data[train_data[label_col] == 0].index.tolist()
        if len(pos_indices) > len(neg_indices):
            sampled_pos_indices = sample(pos_indices, len(neg_indices))
            train_data = train_data.loc[neg_indices + sampled_pos_indices]
        else:
            sampled_neg_indices = sample(neg_indices, len(pos_indices))
            train_data = train_data.loc[pos_indices + sampled_neg_indices]
        return train_data

    train = balance_classes(train)
    train_ae = balance_classes(train_ae)
    train_dmde = balance_classes(train_dmde)

    # Associated features
    X_train, y_train = train.drop(['target'], axis=1), train['target']
    X_test, y_test = test.drop(['target'], axis=1), test['target']
    X_train_ae, y_train_ae = train_ae.drop(['target'], axis=1), train_ae['target']
    X_test_ae, y_test_ae = test_ae.drop(['target'], axis=1), test_ae['target']
    X_train_dmde, y_train_dmde = train_dmde.drop(['target'], axis=1), train_dmde['target']
    X_test_dmde, y_test_dmde = test_dmde.drop(['target'], axis=1), test_dmde['target']

    # Ensure columns in train and test sets match
    common_columns = list(set(X_train.columns) & set(X_test.columns) & set(X_train_ae.columns) & set(X_test_ae.columns) & set(X_train_dmde.columns) & set(X_test_dmde.columns))
    X_train = X_train[common_columns]
    X_test = X_test[common_columns]
    X_train_ae = X_train_ae[common_columns]
    X_test_ae = X_test_ae[common_columns]
    X_train_dmde = X_train_dmde[common_columns]
    X_test_dmde = X_test_dmde[common_columns]

    # Classifier
    n_estimator = 10
    cls = GradientBoostingClassifier(n_estimators=n_estimator)
    cls.fit(X_train, y_train)
    y_pred_rf = cls.predict_proba(X_test)[:, 1]

    # Predictions for ae and dmde
    y_pred_rf_ae = cls.predict_proba(X_test_ae)[:, 1]
    y_pred_rf_dmde = cls.predict_proba(X_test_dmde)[:, 1]

    draw(y_test, y_pred_rf, save_path, 'original')
    draw(y_test_ae, y_pred_rf_ae, save_path, 'Autoencoder')
    draw(y_test_dmde, y_pred_rf_dmde, save_path, 'DPDCA')

# 调用函数来处理"decode data"和"original data"
train_and_evaluate('original_data.csv', 'ae_data.csv', 'dm_de_data.csv')