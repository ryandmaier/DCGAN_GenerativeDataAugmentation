
import json
import datetime

import numpy as np

def save_results(results_file, results):
    all_results = {}
    fname = results_file # '../../../../data3/xray_data_augmentation/test_data/model_results5.json'
    with open(fname,'r') as file: # 12:36 
        try:
            all_results = json.load(file)
        except json.JSONDecodeError as e:
            all_results = {}
        all_results[str(datetime.datetime.now())] = results
    with open(fname,'w') as file:
        # print('all_results: ',all_results)
        print("Saving results to ",fname,str(datetime.datetime.now()))
        json.dump(all_results, file)
    return

def scores(y_pred, y_test):
    # Normal is 1, Pneumonia is 0
    TP = int(np.sum((y_pred > 0.5) & (y_test > 0.5)))
    TN = int(np.sum((y_pred < 0.5) & (y_test < 0.5)))
    FP = int(np.sum((y_pred > 0.5) & (y_test < 0.5)))
    FN = int(np.sum((y_pred < 0.5) & (y_test > 0.5)))
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy: {accuracy}")
    # Precision
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    # Recall (Sensitivity or True Positive Rate)
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    result = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }
    print(result)
    return result
    
def get_scores(model, n, generator, title, batch_size):
    print(f"Testing with {title}")
    labels = []
    for i in range(0, n//batch_size):
        labels.extend(generator[i][1])
    labels = np.array(labels)
    preds = np.array(model.predict(generator))
    print(f"labels shape: {labels.shape}, preds shape: {preds.shape}")
    return scores(preds[:len(labels),0], labels[:,0])
    # save_results(title, result)