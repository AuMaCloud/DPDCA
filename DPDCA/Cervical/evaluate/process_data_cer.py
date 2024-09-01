# -*- coding: utf-8 -*-
import pandas as pd
import sklearn.model_selection as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer

# Read data
DATASETDIR = os.path.expanduser('../data')
df = pd.read_csv(os.path.join(DATASETDIR, 'cervical.csv'), sep=',')

# Handle missing values by imputing
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Train/test split
train_df, test_df = skl.train_test_split(df, test_size=0.2, stratify=df['Biopsy'])

# Report positive labels distribution
print('Positive label percentage in train set', train_df['Biopsy'].sum() / len(train_df))
print('Positive label percentage in test set', test_df['Biopsy'].sum() / len(test_df))

# Associated features
X_train, y_train = train_df.drop(['Biopsy'], axis=1), train_df['Biopsy']
X_test, y_test = test_df.drop(['Biopsy'], axis=1), test_df['Biopsy']

# Scaling features
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Unified sets
train = np.concatenate((X_train, y_train[:, None]), axis=1)
test = np.concatenate((X_test, y_test[:, None]), axis=1)

# Save the processed data
pd.DataFrame(train).to_csv(os.path.join(DATASETDIR, 'train.csv'), index=False)
pd.DataFrame(test).to_csv(os.path.join(DATASETDIR, 'test.csv'), index=False)

def merge_datasets_my():
    # Read the first dataset
    dataset1 = pd.read_csv("../data/train.csv")

    # Read the second dataset
    dataset2 = pd.read_csv("../data/test.csv")

    # Merge the two datasets, assuming they have the same columns
    merged_dataset = pd.concat([dataset1, dataset2], axis=0, ignore_index=True)

    # Drop the first column
    merged_dataset = merged_dataset.iloc[:, 1:]

    # Shuffle the dataset
    shuffled_dataset = shuffle(merged_dataset)

    # Save the merged dataset to an output file
    shuffled_dataset.to_csv("../data/original_data.csv", index=False)

merge_datasets_my()

###############################
######## Classifier ###########
###############################

# Supervised transformation based on random forests
n_estimator = 100
cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
# cls = GradientBoostingClassifier(n_estimators=n_estimator)
# cls = xgb.XGBClassifier(n_estimators=n_estimator)
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)

c_report = metrics.classification_report(y_test, y_pred)
print('Classification report:\n', c_report)

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

# Predict probabilities
y_pred_proba = cls.predict_proba(X_test)[:, 1]

# ROC
fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(y_test, y_pred_proba)
print('AUROC: ', metrics.auc(fpr_rf_lm, tpr_rf_lm))

# PR
precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba)
AUPRC = metrics.auc(recall, precision)
print('AP: ', metrics.average_precision_score(y_test, y_pred_proba))
print('Area under the precision recall curve: ', AUPRC)

# Cross validation
# see https://scikit-learn.org/stable/modules/model_evaluation.html
X, Y = df.drop(['Biopsy'], axis=1), df['Biopsy']
kfold = model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(cls, X, Y, cv=kfold, scoring='roc_auc')
print('AUROC mean: {} std: {}'.format(results.mean(), results.std()))
