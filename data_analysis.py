#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Arjun Thangaraju'
# ---------------------------------------------------------------------------
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import time

start_time = time.time()


with open("saved_pickle_file.pickle", 'rb') as f:
     Test_Data_Time_Series_Error_Val = pickle.load(f)

Test_Data_Time_Series_Error_Val_Array = np.asarray(Test_Data_Time_Series_Error_Val[0])
Test_Data_Time_Series_Error_Val_Array_Combined = np.concatenate(Test_Data_Time_Series_Error_Val[0])
test_data_2_good_val_index = 16000
y_pos_size = len(Test_Data_Time_Series_Error_Val_Array[0]) + test_data_2_good_val_index
y_neg_size = (len(Test_Data_Time_Series_Error_Val_Array[1]) + len(Test_Data_Time_Series_Error_Val_Array[2]) + len(Test_Data_Time_Series_Error_Val_Array[3])) - test_data_2_good_val_index
y_actual = np.concatenate([np.zeros(y_pos_size), np.ones(y_neg_size)])

# Define Min and Max Range
sse_min = min(Test_Data_Time_Series_Error_Val_Array_Combined)
sse_max = max(Test_Data_Time_Series_Error_Val_Array_Combined)
step_val = 0.5
sse_range = int((sse_max-sse_min)/step_val)
my_thresholds_list = np.zeros(sse_range)
TP_list = np.zeros(sse_range)
FP_list = np.zeros(sse_range)
FN_list = np.zeros(sse_range)
TN_list = np.zeros(sse_range)

accuracy_list = np.zeros(sse_range)
precision_list = np.zeros(sse_range)
recall_list = np.zeros(sse_range)

fpr_list = np.zeros(sse_range)
tpr_list = np.zeros(sse_range)


thresholds_list = np.zeros(sse_range)

auc_list = np.zeros(sse_range)

# Loop 1: Loop through threshold variable - range(min(SSE), max(SSE), 0.1)
count = 0
for threshold_index in range(int(sse_min), sse_range, 1):
    print("Loop Number:", count)
    count += 1
    y_pred = np.zeros(len(y_actual))
    threshold_val = sse_min + (threshold_index*step_val)
    my_thresholds_list[threshold_index] = threshold_val
    # Loop 2:
    positive_case_indices = np.argwhere(Test_Data_Time_Series_Error_Val_Array_Combined > threshold_val)
    y_pred[positive_case_indices] = 1
    conf_matrix = confusion_matrix(y_actual, y_pred)
    TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]
    TP = conf_matrix[1,1]

    accuracy = (TP+TN)/(TP+FP+FN+TN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    TP_list[threshold_index] = TP
    FP_list[threshold_index] = FP
    FN_list[threshold_index] = FN
    TN_list[threshold_index] = TN

    accuracy_list[threshold_index] = accuracy
    precision_list[threshold_index] = precision
    recall_list[threshold_index] = recall

fpr_list, tpr_list, thresholds_list = metrics.roc_curve(y_actual, Test_Data_Time_Series_Error_Val_Array_Combined, pos_label=1)
auc_val = metrics.auc(fpr_list, tpr_list)

# indices = np.arange(int(sse_min), sse_range, 1)

fig, ax = plt.subplots()
plt.title('Accuracy, Precision and Recall Curves')
plt.plot(my_thresholds_list, accuracy_list, label='Accuracy')
plt.plot(my_thresholds_list, precision_list, label='Precision')
plt.plot(my_thresholds_list, recall_list, label='Recall')
plt.xlabel('Threshold Values')
plt.ylabel('Metric Values')
plt.legend()
plt.show()

fig, ax = plt.subplots()

plt.plot(fpr_list, tpr_list)
plt.plot(my_thresholds_list/max(my_thresholds_list), my_thresholds_list/max(my_thresholds_list), '--r')

plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(["AUC=%.4f" % auc_val])
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
