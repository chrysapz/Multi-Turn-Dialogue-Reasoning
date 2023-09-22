import torch
import numpy as np
import datetime

# see https://github.com/Nealcly/MuTual/blob/master/eval_sample/eval.py
def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_checkpoint_name(config):
    now = datetime.datetime.now()
    date_info = f'{now.month}_{now.day}_{now.hour}_{now.minute}'
    config_name =  f"{config['dataset_name']}_{config['model_name']}_{date_info}"

    print('date info ', date_info)
    return config_name

def calculate_IR_metrics(sorted_grouped_data, labeled_data):
    counter_r1, counter_r2, mrr = 0, 0, 0
    for sentence_id in sorted_grouped_data:
        golden = labeled_data[sentence_id]
        output_list = sorted_grouped_data[sentence_id]
        assert sorted(output_list) == [0,1,2,3]
        index = output_list.index(golden)
        if index == 0:
            counter_r1 += 1
        elif index == 1:
            counter_r2 += 1
        mrr += 1 / (index + 1)
    
    count_data = len(sorted_grouped_data)
    mrr = mrr/count_data
    r_1 = counter_r1 / count_data
    r_2 = (counter_r2 +counter_r1) / count_data
    print("R@1: %.3f \t R@2: %.3f \t MRR %.3f" %(r_1, r_2, mrr))
    return r_1, r_2, mrr

#todo dataset cartography

# just tests
if __name__=='__main__':
    sorted_grouped_data = {1:[2,0,1,3],2:[1,2,0,3]}
    labeled_data = {1:2,2:1}
    r_1, r_2, mrr = calculate_IR_metrics(sorted_grouped_data, labeled_data)
    assert(r_1 == r_2== mrr == 1.0)

    sorted_grouped_data = {1:[2,0,1,3],2:[1,2,0,3]}
    labeled_data = {1:2,2:0}
    r_1, r_2, mrr = calculate_IR_metrics(sorted_grouped_data, labeled_data)
    assert(r_1 == 1/2 == r_2)
    assert(mrr == 2/3)
