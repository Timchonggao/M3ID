import csv
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV

from IPython.display import display 
import plotly.express as px 


import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_probability_distribution_with_misclassification_rate(y_test, y_pred_proba, bin_width=0.05):
    """
    Plots a combined histogram showing the distribution of predicted probabilities for Label 1.
    Correctly classified and misclassified samples are shown in different colors.
    Additionally, plots a line indicating the misclassification rate (%) across probability bins.

    Args:
        y_test (numpy.ndarray or torch.Tensor): True labels of shape (N,).
        y_pred_proba (numpy.ndarray or torch.Tensor): Predicted probabilities for Label 1 of shape (N,).
        bin_width (float, optional): Width of each bin in the histogram. Default is 0.05.
    """

    # Helper function to convert tensors to numpy arrays
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Input data must be a torch.Tensor or numpy.ndarray.")

    # Convert inputs to numpy arrays
    y_test = to_numpy(y_test)
    y_pred_proba = to_numpy(y_pred_proba)

    # Ensure y_test and y_pred_proba are 1-D arrays
    y_test = y_test.flatten()
    y_pred_proba = y_pred_proba.flatten()

    # **Validation Checks:**
    if y_test.shape[0] != y_pred_proba.shape[0]:
        raise ValueError(f"y_test and y_pred_proba must have the same number of samples. "
                         f"Got {y_test.shape[0]} and {y_pred_proba.shape[0]} respectively.")

    # Ensure y_test contains only binary labels 0 and 1
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("y_test must contain only binary labels 0 and 1.")

    # **Clip probabilities to [0, 1] to avoid anomalies**
    y_pred_proba = np.clip(y_pred_proba, 0, 1)

    # **Compute Predicted Labels:**
    # Using 0.5 as the threshold. Adjust if a different threshold is required.
    y_pred_label = (y_pred_proba >= 0.5).astype(int)

    # **Identify Correctly Classified and Misclassified Samples:**
    correct = (y_pred_label == y_test)
    misclassified = ~correct  # Equivalent to y_pred_label != y_test

    # **Extract prob_label1 for both correct and misclassified samples**
    prob_correct = y_pred_proba[correct]
    prob_misclassified = y_pred_proba[misclassified]

    # **Define Histogram Bins:**
    bins = np.arange(0, 1 + bin_width, bin_width)  # e.g., [0.00, 0.05, 0.10, ..., 1.00]

    # **Compute Histogram Counts:**
    counts_correct, _ = np.histogram(prob_correct, bins=bins)
    counts_misclassified, _ = np.histogram(prob_misclassified, bins=bins)
    total_counts = counts_correct + counts_misclassified

    # **Calculate Misclassification Rate per Bin:**
    # To avoid division by zero, use np.where
    misclass_rate = np.where(total_counts > 0, counts_misclassified / total_counts * 100, 0)

    # **Define Bin Centers for Plotting the Line:**
    bin_centers = bins[:-1] + bin_width / 2

    # **Plot Combined Histogram and Misclassification Rate Line:**
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot correctly classified samples
    ax1.hist(prob_correct, bins=bins, alpha=0.6, label='Correctly Classified',
             color='green', edgecolor='black')

    # Plot misclassified samples
    ax1.hist(prob_misclassified, bins=bins, alpha=0.6, label='Misclassified',
             color='red', edgecolor='black')

    # Set labels and title for the first y-axis
    ax1.set_xlabel('Probability of Label 1')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Probability Distribution and Misclassification Rate')
    ax1.set_xticks(bins)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.75)

    # **Plot Misclassification Rate Line:**
    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.plot(bin_centers, misclass_rate, color='blue', marker='o', linestyle='-',
             linewidth=2, label='Misclassification Rate (%)')

    # Set labels and limits for the second y-axis
    ax2.set_ylabel('Misclassification Rate (%)')
    ax2.set_ylim(0, 100)  # Percentages from 0% to 100%
    ax2.legend(loc='upper right')

    # **Final Layout Adjustments:**
    plt.tight_layout()
    plt.show()
    plt.savefig('misclassification_rate.png')  # Uncomment to save the plot as a file
    plt.close()  # Uncomment to close the plot window after saving

def plot_probability_distribution(train_data, val_data, test_data, bin_width=0.1):
    """
    Plots a combined histogram showing the distribution of probabilities for Label 1.
    Correctly classified and misclassified samples are shown in different colors.
    Additionally, plots a line indicating the misclassification rate per probability bin.

    Args:
        train_data (torch.Tensor or numpy.ndarray): Training data with probabilities and labels.
        val_data (torch.Tensor or numpy.ndarray): Validation data with probabilities and labels.
        test_data (torch.Tensor or numpy.ndarray): Test data with probabilities and labels.
        bin_width (float): Width of each bin in the histogram. Default is 0.05.
    """

    # Helper function to convert tensors to numpy arrays
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Input data must be a torch.Tensor or numpy.ndarray.")

    # Convert datasets to numpy arrays
    train_data = to_numpy(train_data)
    val_data = to_numpy(val_data)
    test_data = to_numpy(test_data)

    # Concatenate all datasets if needed; in user code, it's using test_data only
    # Uncomment the following line to use all datasets
    # all_data = np.concatenate((train_data, val_data, test_data), axis=0)

    all_data = test_data  # Using only test_data as per user code

    # **Assumptions on Data Structure:**
    # - Column -3: Probability of Label 0 (prob_label0)
    # - Column -2: Probability of Label 1 (prob_label1)
    # - Column -1: True Label (label_true)

    # Extract probabilities and true labels
    prob_label0 = all_data[:, -3]
    prob_label1 = all_data[:, -2]
    label_true = all_data[:, -1]

    # **Validation:**
    # Ensure that probabilities are within [0, 1]
    prob_label0 = np.clip(prob_label0, 0, 1)
    prob_label1 = np.clip(prob_label1, 0, 1)

    # **Compute Predicted Labels:**
    # If prob_label1 > prob_label0, predict Label 1; else, predict Label 0
    predicted_labels = np.where(prob_label1 > prob_label0, 1, 0)

    # **Identify Correctly Classified and Misclassified Samples:**
    correct = predicted_labels == label_true
    misclassified = ~correct  # Alternatively: predicted_labels != label_true

    # **Extract prob_label1 for both correct and misclassified samples**
    prob_correct = prob_label1[correct]
    prob_misclassified = prob_label1[misclassified]

    # **Define Histogram Bins:**
    bins = np.arange(0, 1 + bin_width, bin_width)  # e.g., [0.00, 0.05, 0.10, ..., 1.00]

    # **Compute Histogram Counts:**
    counts_correct, _ = np.histogram(prob_correct, bins=bins)
    counts_misclassified, _ = np.histogram(prob_misclassified, bins=bins)
    total_counts = counts_correct + counts_misclassified

    # **Calculate Misclassification Rate per Bin:**
    # To avoid division by zero, use np.where
    misclass_rate = np.where(total_counts > 0, counts_misclassified / total_counts * 100, 0)

    # **Define Bin Centers for Plotting the Line:**
    bin_centers = bins[:-1] + bin_width / 2

    # **Plot Combined Histogram:**
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot correctly classified samples
    ax1.hist(prob_correct, bins=bins, alpha=0.6, label='Correctly Classified', color='green', edgecolor='black')

    # Plot misclassified samples
    ax1.hist(prob_misclassified, bins=bins, alpha=0.6, label='Misclassified', color='red', edgecolor='black')

    ax1.set_xlabel('Probability of Label 1')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Probability Distribution and Misclassification Rate')
    ax1.set_xticks(bins)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.75)

    # **Plot Misclassification Rate Line:**
    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.plot(bin_centers, misclass_rate, color='blue', marker='o', linestyle='-', linewidth=2,
             label='Misclassification Rate (%)')
    ax2.set_ylabel('Misclassification Rate (%)')
    ax2.set_ylim(0, 100)  # Percentages from 0% to 100%
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    save_path = 'probability_distribution.png'
    plt.savefig(save_path, dpi=300)

Acc = []
Presicion = [] 
Recall = [] 
F1 = [] 
Auc = []
mean = []
std = []

for fold in range(5):

    train_data = np.load('/data3/gaochong/project/M3ID/static_info/all_metrics_fold{}_train.npy'.format(fold))
    val_data = np.load('/data3/gaochong/project/M3ID/static_info/all_metrics_fold{}_val.npy'.format(fold))
    test_data = np.load('/data3/gaochong/project/M3ID/static_info/all_metrics_fold{}_test.npy'.format(fold))

# acc,presicion,recall,F1,AUC,swinunet-ce-lr0.001
# 0.7507,0.7458,0.639,0.6871,0.8445
# 0.0049,0.0274,0.0293,0.0084,0.0041

    # X_train = train_data[:, -3:-1]
    # X_val = val_data[:, -3:-1]
    # X_test = test_data[:, -3:-1]
# mean acc:  0.7687296416938111
# mean presicion:  0.6864803195324741
# mean recall:  0.7347457627118643
# mean F1:  0.7096130993345434
# mean AUC:  0.8435790512061698

    X_train = train_data[:, 47:-3]
    X_val = val_data[:, 47:-3]
    X_test = test_data[:, 47:-3]
# mean acc:  0.6335504885993485
# mean presicion:  0.51927342926825
# mean recall:  0.6474576271186441
# mean F1:  0.5750433251630193
# mean AUC:  0.6936911487758946

#     X_train = train_data[:, 47:-1]
#     X_val = val_data[:, 47:-1]
#     X_test = test_data[:, 47:-1]
# mean acc:  0.7596091205211726
# mean presicion:  0.6732461562901398
# mean recall:  0.7296610169491525
# mean F1:  0.7000396464903094
# mean AUC:  0.8378262039278989

    # X_train = train_data[:, 47:-1]
    # X_train = np.delete(X_train, [21, 43], axis=1)
    # X_val = val_data[:, 47:-1]
    # X_val = np.delete(X_val, [21, 43], axis=1)
    # X_test = test_data[:, 47:-1]
    # X_test = np.delete(X_test, [21, 43], axis=1)
# mean acc:  0.7644951140065147
# mean presicion:  0.6794389379683498
# mean recall:  0.7347457627118643
# mean F1:  0.7057517724774338
# mean AUC:  0.8380436732131648


    X_train_normal = X_train
    X_val_normal = X_val
    X_test_normal = X_test

    if np.sum(np.isnan(X_train_normal)) > 0 or np.sum(np.isnan(X_val_normal)) > 0 or np.sum(np.isnan(X_test_normal)) > 0:
        print('nan detected')
        continue

    y_train = train_data[:, -1]
    y_val = val_data[:, -1]
    y_test = test_data[:, -1]
    plot_probability_distribution(train_data, val_data, test_data)
    catboost_model = CatBoostClassifier(iterations=1000, auto_class_weights='SqrtBalanced', depth=10, learning_rate=0.01, l2_leaf_reg=10, verbose=True, loss_function='Logloss', eval_metric='AUC', random_seed=1234)

    catboost_model.fit(X_train_normal, y_train, eval_set=(X_val_normal, y_val), early_stopping_rounds=300, use_best_model=True, verbose=10)

    #feature importance 
    dfimportance = catboost_model.get_feature_importance(prettified=True) 
    dfimportance = dfimportance.sort_values(by = "Importances").iloc[:]
    display(dfimportance)

    y_pred_train = catboost_model.predict(X_train_normal)
#     print("y_pred_train: ", y_pred_train)
    # 计算训练集上的准确率
    accuracy_train = accuracy_score(y_train, y_pred_train)
    # 计算精确度
    precision_train = precision_score(y_train, y_pred_train)
    # 计算召回率
    recall_train = recall_score(y_train, y_pred_train)
    # 计算F1分数
    f1_train = f1_score(y_train, y_pred_train)
    # 计算AUC值
    y_pred_proba_train = catboost_model.predict_proba(X_train_normal)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_pred_proba_train)
    print('train fold %d acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
          (fold, accuracy_train, roc_auc_train, f1_train, precision_train, recall_train))
    
    # 预测验证集
    y_pred_val = catboost_model.predict(X_val_normal)
    # 计算验证集上的准确率
    accuracy_val = accuracy_score(y_val, y_pred_val)
    # 计算精确度
    precision_val = precision_score(y_val, y_pred_val)
    # 计算召回率
    recall_val = recall_score(y_val, y_pred_val)
    # 计算F1分数
    f1_val = f1_score(y_val, y_pred_val)
    # 计算AUC值
    y_pred_proba_val = catboost_model.predict_proba(X_val_normal)[:, 1]
    roc_auc_val = roc_auc_score(y_val, y_pred_proba_val)
    print('val fold %d acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
          (fold, accuracy_val, roc_auc_val, f1_val, precision_val, recall_val)) 

    # 预测测试集
    y_pred_test = catboost_model.predict(X_test_normal)
    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred_test)
    # 计算精确度
    precision = precision_score(y_test, y_pred_test)
    # 计算召回率
    recall = recall_score(y_test, y_pred_test)
    # 计算F1分数
    f1 = f1_score(y_test, y_pred_test)
    # 计算AUC值
    y_pred_proba = catboost_model.predict_proba(X_test_normal)[:, 1]
    print(y_pred_proba.shape)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plot_probability_distribution_with_misclassification_rate(y_test, y_pred_proba)
    Acc.append(accuracy)
    Presicion.append(precision)
    Recall.append(recall)
    F1.append(f1)
    Auc.append(roc_auc)

    print('test fold %d acc: %.4f, auc: %.4f, f1: %.4f, presicion: %.4f, recall: %.4f \n' % \
          (fold, accuracy, roc_auc, f1, precision, recall))  

csvFile = open("../results-test.csv", "a+")
writer = csv.writer(csvFile)
name = 'swinunet_cis_catboost'
head = ["acc", "presicion", "recall", "F1", "AUC", name]
writer.writerow(head)

mean.append(round(np.mean(Acc),4))
mean.append(round(np.mean(Presicion),4))
mean.append(round(np.mean(Recall),4))
mean.append(round(np.mean(F1),4))
mean.append(round(np.mean(Auc),4))
writer.writerow(mean)

std.append(round(np.std(Acc),4))
std.append(round(np.std(Presicion),4))
std.append(round(np.std(recall),4))
std.append(round(np.std(F1),4))
std.append(round(np.std(Auc),4))
writer.writerow(std)

csvFile.close()

print("mean acc: ", np.mean(Acc))
print("mean presicion: ", np.mean(Presicion))
print("mean recall: ", np.mean(Recall))
print("mean F1: ", np.mean(F1))
print("mean AUC: ", np.mean(Auc))