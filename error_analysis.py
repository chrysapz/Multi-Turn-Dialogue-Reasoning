from utils import load_pickle, create_dicts_from_tuples, calculate_IR_metrics, convert_id2label_id_to_element
import os
from data import load_all_samples
from evaluate import sort_grouped_data, group_data
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

def main():
    labeled_data = load_pickle('dict_probs/labeled_data.pkl')

    runs = ['augment_final_finetuned_sim','augment_final_finetuned_gold','baseline','repeat_gold_final_finetuned','repeat_sim_final_finetuned']
    for run in runs:
        print(run)
        name = os.path.join("dict_probs",run,'dict_probs.pkl')
        with open(name, 'rb') as f:
            predictions = pickle.load(f)
  
            sorted_data = sort_grouped_data(predictions)
            r_1, r_2, mrr = calculate_IR_metrics(sorted_data, labeled_data)

            differences, pred_arr = check_when_correct(predictions, labeled_data) # examine what is happening in TPs
            labels, preds, diff = check_correct_pred(predictions, labeled_data) # examine even when it is wrong
            # calculate roc-auc
            print("ROC-AUC Score: ",roc_auc_score(labels, preds))

# code for checking for correct

def check_correct_pred(predictions, actual):
    lab_arr = []
    pred_arr = []
    diff = []

    for instance_id, preds in predictions.items():
        # Extracting the predicted confidence for the actual class of the instance
        actual_label = actual[instance_id]
        pred = preds[actual_label][1]
        
        preds = [i[1] for i in preds]
        ind = np.argmax(preds)

        if actual_label == ind:
            lab = 1
        else:
            lab = 0
            max_pred = preds[ind]
            diff.append(abs(max_pred-pred)) # check when not correct how far is from correct

        lab_arr.append(lab)
        pred_arr.append(pred)
        
    return lab_arr, pred_arr, diff

# code for checking for correct
def check_when_correct(predictions, actual):
    differences = []
    pred_arr = []

    for instance_id, preds in predictions.items():
        # Extracting the predicted confidence for the actual class of the instance
        actual_label = actual[instance_id]
        preds = [i[1] for i in preds]
        ind = np.argmax(preds)

        if actual_label == ind:
            pred = preds[int(actual_label)]
            differences.append(1-pred)
            pred_arr.append(pred)
    return differences, pred_arr



if __name__ == "__main__":
    main()