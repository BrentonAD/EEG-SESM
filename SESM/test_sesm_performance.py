import os
import pickle
import json
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from sesm import get_data
from trainer import PLModel

EXPERIMENT_ID = 41

CLASS_DICT = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

output_location = 'test_outputs'

def get_freer_gpu():
    # Run nvidia-smi command to get GPU memory info
    command = "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits"
    output = os.popen(command).read().strip().split('\n')

    # Parse the output to find GPU with most free memory
    max_memory = -1
    max_memory_index = -1
    for index, line in enumerate(output):
        free_memory, _ = map(int, line.split(', '))
        if free_memory > max_memory:
            max_memory = free_memory
            max_memory_index = index

    return max_memory_index

def main():
    free_gpu_id = get_freer_gpu()
    #free_gpu_id=0
    print("select gpu:", free_gpu_id)
    torch.cuda.set_device(torch.device(free_gpu_id))

    config = json.load(open(f"configs/sleep_edf_exp{EXPERIMENT_ID}.json", "r"))
    #config = json.load(open(f"configs/sleep_edf_exp37-40.json", "r"))
    train_loader, val_loader, test_loader, class_weights, max_len = get_data(
        config["dataset"], "E:\s222165064", config["batch_size"],
        val_subject_ids=[0,1],
        test_subject_ids=[2,7,12]
    )
    config.update({"class_weights": class_weights, "max_len": max_len})

    plmodel = PLModel(stage=2, **config)
    plmodel.model.load_state_dict(torch.load(f"models/trained_model_exp{EXPERIMENT_ID}.pt"))

    model = plmodel.model
    model.eval()
    model.training

    # Calculate Accuracy
    print("Calculating Accuracy...")
    y_true = []
    y_pred = []

    for x, y in test_loader:
        y_hat, _, _, _ = model(x)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(y_hat.argmax(1).numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    os.makedirs(output_location, exist_ok=True)
    os.makedirs(f"{output_location}/exp{EXPERIMENT_ID}", exist_ok=True)

    with open(f'{output_location}/exp{EXPERIMENT_ID}/y_true_test.pkl','wb') as f:
        pickle.dump(y_true, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{output_location}/exp{EXPERIMENT_ID}/y_pred_test.pkl','wb') as f:
        pickle.dump(y_pred, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate AOPC
    print("Calculating AOPC...")
    AOPC = torch.zeros(config['n_heads'])
    total = 0

    for x,y in test_loader:
        y_hat_, _, selective_actions, relevance_weights = model(x, x != 0,ignore_relevance_weights=True) 
        batch_size = x.shape[0]
        
        relevance_weights = relevance_weights.detach().cpu()
        y_hat_ = y_hat_.detach().reshape(batch_size, config['n_heads'], -1).cpu()
        
        attn_arg = torch.sort(relevance_weights, 1, descending=True)[1] 
        
        attn_logits = y_hat_ * relevance_weights.unsqueeze(-1)
        
        for i in range(config['n_heads']):
            AOPC[i] += F.softmax(attn_logits[:, i:].sum(1), -1)[torch.arange(batch_size).long(), y.cpu().long()].sum()
            
        total += batch_size

        del x
        del y
        del y_hat_
        del relevance_weights
        del attn_arg
        del attn_logits
        # break

    os.makedirs(output_location, exist_ok=True)
    os.makedirs(f"{output_location}/exp{EXPERIMENT_ID}", exist_ok=True)

    with open(f'{output_location}/exp{EXPERIMENT_ID}/AOPC.pkl','wb') as f:
        pickle.dump(AOPC, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{output_location}/exp{EXPERIMENT_ID}/total.pkl','wb') as f:
        pickle.dump(total, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()