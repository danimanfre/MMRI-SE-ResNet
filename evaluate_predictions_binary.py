import pickle
import torch
import seaborn as sns
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import f1_score

def plot_confMat(target, predicted, seed, out_parentDir="./confusion_matrix_result"):
    labels=["negative", "positive"]
    """
    target of shape: (# of samples,)
    predicted of shape: (# of samples,)
    """
    plt.figure(figsize=(5, 5))
    cm = metrics.confusion_matrix(target, predicted, labels=list(range(2)))
    cm_ratio = cm.astype("float") / cm.sum(axis=1).reshape(-1, 1)
    df_cm = pd.DataFrame(cm_ratio, index=labels, columns=labels)

    sns.heatmap(df_cm, annot=cm, fmt="d", cmap="Blues", vmax=1, vmin=0, annot_kws={"fontsize": 20})
    plt.xlabel("Predicted label", fontsize=18), plt.ylabel("True label", fontsize=18), plt.yticks(rotation=0)
    plt.savefig(f"{out_parentDir}/confusion_matrix_notse_binary_aggregate_" + str(seed) +".png", bbox_inches="tight", pad_inches=0)
    print(f"You can find the confusion matrix following this path {out_parentDir}/confusion_matrix_notse_binary_aggregate.png")

# Load the two hashmaps from the file
seed = 54665
filename = "my_predictions" + str(seed) + ".pickle"
with open(filename, "rb") as file:
    accumulate_logits, map_true_label = pickle.load(file)
print(seed)

number_of_slices = 120
threshold = 0.5

# Simple Majority Voting -> The most frequent class is the final prediction

accumulate_predictions = {}
counts = {}
most_frequent_value = {}
unique_values = torch.tensor([0, 1])
for id in accumulate_logits.keys():
    accumulate_predictions[id] = torch.tensor((accumulate_logits[id] >= threshold)).float()

    # Count the occurrences of each unique value
    _, counts[id] = accumulate_predictions[id].view(-1).unique(return_counts=True)
    
    
    # Find the most frequent value
    most_frequent_value[id] = counts[id].argmax().item()

    counts[id] = counts[id] * 100.0 / number_of_slices



corret_predictions_true = 0
corret_predictions_false = 0
corret_predictions = 0
y_true = []
y_pred = []

for id in most_frequent_value.keys():

    predicted_value = most_frequent_value[id]
    true_label = map_true_label[id]
    if (predicted_value == true_label):
        corret_predictions += 1
        if(true_label):
            corret_predictions_true +=1
        else:
            corret_predictions_false += 1

    # Build an array of true labels and predicted labels for computing Macro F1
    y_true.append(true_label)
    y_pred.append(predicted_value)


# Print statistics
print('###   Simple Majority Voting Evaluation   ###')
print(f"These are the F1 Macro value: {f1_score(y_true, y_pred, average='macro')}")
print(f"These are the accuracy value: {float(corret_predictions)/len(map_true_label)}")

count_0 = 0
count_1 = 0

# Iterate through the dictionary values
for value in map_true_label.values():
    if value == 0:
        count_0 += 1
    elif value == 1:
        count_1 += 1

# Print the counts
print("Number of occurrences with 0:", count_0)
print("Number of occurrences with 1:", count_1)
print("Number of correct predictions:", corret_predictions)
print("Number of positive correct predictions:", corret_predictions_true* 100.0 / count_1)
print("Number of negative correct predictions:", corret_predictions_false* 100.0 / count_0)


# Simple Mean #

mean_predictions = {}

predictions = {}

corret_predictions_true = 0
corret_predictions_false = 0
corret_predictions = 0
y_true = []
y_pred = []

for id in accumulate_logits.keys():
    
    mean_predictions[id] = torch.tensor(accumulate_logits[id]).mean()

    predictions[id] = (mean_predictions[id] >= threshold).float()

    predicted_value = predictions[id]
    true_label = map_true_label[id]
    if (predicted_value == true_label):
        corret_predictions += 1
        if(true_label):
            corret_predictions_true +=1
        else:
            corret_predictions_false += 1

    # Build an array of true labels and predicted labels for computing Macro F1
    y_true.append(true_label)
    y_pred.append(predicted_value)


# Print statistics
print('###   Simple Mean Evaluation   ###')
print(f"These are the F1 Macro value: {f1_score(y_true, y_pred, average='macro')*100 :.2f}%")
print(f"These are the accuracy value: {float(corret_predictions)/len(map_true_label)*100 :.2f}%")

count_0 = 0
count_1 = 0

# Iterate through the dictionary values
for value in map_true_label.values():
    if value == 0:
        count_0 += 1
    elif value == 1:
        count_1 += 1

# Print the counts
print("Number of negative samples:", count_0)
print("Number of positive samples:", count_1)
print("Number of correct predictions:", corret_predictions)
print(f"Percentage of positive correct predictions: {corret_predictions_true* 100.0 / count_1:.2f}%")
print(f"Percentage of negative correct predictions: {corret_predictions_false* 100.0 / count_0:.2f}%")

plot_confMat(y_true, y_pred, seed)
