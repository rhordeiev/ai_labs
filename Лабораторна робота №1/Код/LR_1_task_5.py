import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


df = pd.read_csv('data_metrics.csv')
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')


def hordeiev_find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def hordeiev_find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def hordeiev_find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def hordeiev_find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = hordeiev_find_TP(y_true, y_pred)
    FN = hordeiev_find_FN(y_true, y_pred)
    FP = hordeiev_find_FP(y_true, y_pred)
    TN = hordeiev_find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def hordeiev_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


def hordeiev_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


print('Accuracy RF: %.3f' % (hordeiev_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (hordeiev_accuracy_score(df.actual_label.values, df.predicted_LR.values)))


def hordeiev_recall_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


print('Recall RF: %.3f' % (hordeiev_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (hordeiev_accuracy_score(df.actual_label.values, df.predicted_LR.values)))


def hordeiev_precision_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


print('Precision RF: %.3f' % (hordeiev_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (hordeiev_accuracy_score(df.actual_label.values, df.predicted_LR.values)))


def hordeiev_f1_score(y_true, y_pred):
    # calculates the fraction of samples
    recall = hordeiev_recall_score(y_true, y_pred)
    precision = hordeiev_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


print('F1 RF: %.3f' % (hordeiev_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (hordeiev_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (hordeiev_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (hordeiev_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (hordeiev_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (hordeiev_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print(
    'Accuracy RF: %.3f' % (hordeiev_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (hordeiev_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (
    hordeiev_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (hordeiev_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))


fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

