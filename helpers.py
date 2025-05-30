import os
from pathlib import Path
import random
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import torch.utils.data as data
from irregular_sampled_datasets import PersonData, ETSMnistData, XORData, CustomData 
from ctf4science.data_module import load_dataset, load_validation_dataset

def get_single_file_name(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Check if there is exactly one file in the directory
    if len(files) != 1:
        raise ValueError(f"The directory {directory} does not contain exactly one file.")
    
    # Return the name of the single file
    return files[0]

def forward_model(model, train_mat, timespans, output_timesteps, device):
    model.to(device)
    train_mat = train_mat.to(device)
    timespans = timespans.to(device)

    all_outputs = []
    cur_input = train_mat
    if output_timesteps > 1:
        for i in tqdm(range(output_timesteps), "Unrolling Model"):
            out = model(cur_input, timespans, None) # (1, 3)
            cur_input = torch.concatenate([cur_input[0], out])[1:,:]
            cur_input = cur_input.unsqueeze(0)
            all_outputs.append(out)
        all_outputs_mat = torch.concatenate(all_outputs)
        return all_outputs_mat
    else:
        out = model(cur_input, timespans, None) # (1, 3)
        all_outputs.append(out)
        all_outputs_mat = torch.concatenate(all_outputs)
        return all_outputs_mat


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset_trainer(args):
    if args.dataset == "person":
        dataset = PersonData()
        train_x = torch.Tensor(dataset.train_x)
        train_y = torch.LongTensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_t)
        test_x = torch.Tensor(dataset.test_x)
        test_y = torch.LongTensor(dataset.test_y)
        test_ts = torch.Tensor(dataset.test_t)
        train = data.TensorDataset(train_x, train_ts, train_y)
        test = data.TensorDataset(test_x, test_ts, test_y)
        return_sequences = True
        batch_size = 64
    elif args.dataset in ["ODE_Lorenz", "PDE_KS", "Lorenz_Official", "KS_Official"]:
        return_sequences = False
        dataset = CustomData(args)
        train_x = torch.Tensor(dataset.train_events)
        train_y = torch.Tensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_elapsed)
        train = data.TensorDataset(train_x, train_ts, train_y)
        test = train
        batch_size = args.batch_size
    else:
        if args.dataset == "et_mnist":
            dataset = ETSMnistData(time_major=False)
        elif args.dataset == "xor":
            dataset = XORData(time_major=False, event_based=False, pad_size=32)
        else:
            raise ValueError("Unknown dataset '{}'".format(args.dataset))
        return_sequences = False
        train_x = torch.Tensor(dataset.train_events)
        train_y = torch.LongTensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_elapsed)
        train_mask = torch.Tensor(dataset.train_mask)
        test_x = torch.Tensor(dataset.test_events)
        test_y = torch.LongTensor(dataset.test_y)
        test_ts = torch.Tensor(dataset.test_elapsed)
        test_mask = torch.Tensor(dataset.test_mask)
        train = data.TensorDataset(train_x, train_ts, train_y, train_mask)
        test = data.TensorDataset(test_x, test_ts, test_y, test_mask)
        batch_size = 64
    trainloader = data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)
    in_features = train_x.size(-1)
    num_classes = train_x.shape[2] if args.dataset in ["ODE_Lorenz", "PDE_KS", "KS_Official", "Lorenz_Official"] else int(torch.max(train_y).item() + 1)
    return trainloader, testloader, in_features, num_classes, return_sequences, batch_size


# ODE LSTMS
#train_x.shape
#torch.Size([1900, 100, 3])
#train_ts.shape
#torch.Size([1900, 100, 1])
#train_y.shape
#torch.Size([1900, 3])
#train_mask.shape
#torch.Size([1900, 100, 3])

# XOR
#train_x.shape
#torch.Size([100000, 32, 1])
#train_ts.shape
#torch.Size([100000, 32, 1])
#train_y.shape
#torch.Size([100000])
#train_mask.shape
#torch.Size([100000, 32])