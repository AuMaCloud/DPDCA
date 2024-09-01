# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Read data
DATASETDIR = os.path.expanduser('../data')
df = pd.read_csv(os.path.join(DATASETDIR, 'Epileptic Seizure Recognition.csv'))

# Set seizure activity label with 1 and the others with zero
# Info about initial class labels: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
df['y'] = (df['y'] == 1).replace(True,1).replace(False,0)

# Separate data into positive (y=1) and negative (y=0) samples
df_positive = df[df['y'] == 1]
df_negative = df[df['y'] == 0]

# Associated features
X_df_positive, y_df_positive, X_df_negative, y_df_negative = df_positive.drop(['y','Unnamed'], axis=1), df_positive['y'], df_negative.drop(['y','Unnamed'], axis=1), df_negative['y']

corcof = np.corrcoef(np.transpose(X_df_positive))
ax = sns.heatmap(corcof, annot=False)
plt.show()

# Scaling features
# sc = StandardScaler()
sc = MinMaxScaler()
X_df_positive = sc.fit_transform(X_df_positive)
X_df_negative = sc.transform(X_df_negative)

# unified sets
positive = np.concatenate((X_df_positive, y_df_positive[:,None]), axis=1)
negative = np.concatenate((X_df_negative, y_df_negative[:,None]), axis=1)

# Save the processed data
pd.DataFrame(positive).to_csv(os.path.join(DATASETDIR,'positive.csv'))
pd.DataFrame(negative).to_csv(os.path.join(DATASETDIR,'negative.csv'))


def merge_datasets_my():
    dataset1 = pd.read_csv("../data/positive.csv")

    dataset2 = pd.read_csv("../data/negative.csv")

    merged_dataset = pd.concat([dataset1, dataset2], axis=0, ignore_index=True)

    merged_dataset = merged_dataset.iloc[:, 1:]

    shuffled_dataset = shuffle(merged_dataset)

    shuffled_dataset.to_csv("../data/original_data.csv", index=False)

merge_datasets_my()


# -*- coding: utf-8 -*-
import pandas as pd
import sklearn.model_selection as skl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np
from random import sample
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
DATASETDIR = os.path.expanduser('../data')
df = pd.read_csv(os.path.join(DATASETDIR, 'Epileptic Seizure Recognition.csv'))

# Set seizure activity label with 1 and the others with zero
# Info about initial class labels: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
df['y'] = (df['y'] == 1).replace(True,1).replace(False,0)

# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.2,
                                   stratify = df['y'])


# Report positive labels distribution
print('Positive label percentage in train set', train_df['y'].sum()/len(train_df))
print('Positive label percentage in test set', test_df['y'].sum()/len(test_df))

# make equal number of positive and negative smaples in training
p_inds = train_df[train_df.y==1].index.tolist()
np_inds = train_df[train_df.y==0].index.tolist()

np_sample = sample(np_inds,len(p_inds))
train_df = train_df.loc[p_inds + np_sample]

# Percentage of positive labels in the training
print('Positive label percentage in train set', train_df['y'].sum()/len(train_df))

# Associated features
X_train, y_train, X_test, y_test = train_df.drop(['y','Unnamed'], axis=1), train_df['y'], test_df.drop(['y','Unnamed'], axis=1), test_df['y']

corcof = np.corrcoef(np.transpose(X_train))
ax = sns.heatmap(corcof, annot=False)
plt.show()

# Scaling features
# sc = StandardScaler()
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# unified sets
train = np.concatenate((X_train, y_train[:,None]), axis=1)
test = np.concatenate((X_test, y_test[:,None]), axis=1)

# Save the processed data
pd.DataFrame(train).to_csv(os.path.join(DATASETDIR,'train.csv'))
pd.DataFrame(test).to_csv(os.path.join(DATASETDIR,'test.csv'))

###############################
######## Classifier ###########
###############################

# Supervised transformation based on random forests
# Good to know about feature transformation
n_estimator=10
# cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
cls = GradientBoostingClassifier(n_estimators=n_estimator)
cls.fit(X_train, y_train)
y_pred_rf = cls.predict_proba(X_test)[:, 1]

# ROC
fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(y_test, y_pred_rf)
print('AUROC: ', metrics.auc(fpr_rf_lm,tpr_rf_lm))

# PR
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_rf)
AUPRC = metrics.auc(recall, precision)
print('AP: ', metrics.average_precision_score(y_test, y_pred_rf))
print('Area under the precision recall curve: ', AUPRC)

