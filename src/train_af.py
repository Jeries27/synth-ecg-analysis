import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import json
import torch.nn as nn
from losses import *
from models.SSSD_ECG import SSSD_ECG
from utils import *
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import wfdb
from tqdm import tqdm

from datasets import LongTermDataset

from utils import ECGTransform


def train(
        output_directory,
        ckpt_iter,
        n_iters,
        iters_per_ckpt,
        iters_per_logging,
        learning_rate,
        batch_size,
):
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint,
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """
    local_path = "af_final_no_quantize"
    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available - working on {device}")

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    net = SSSD_ECG(**model_config).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    ######### load checkpoint #########
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
        print(f'{ckpt_iter=}')
    # ckpt_iter = 1
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')

    fs = 128
    folder_path = '/mnt/qnap/jeries/physionet.org/files/ltafdb/1.0.0/'
    record_names = []
    for file in os.listdir(folder_path):
        if file.endswith('.hea'):  # we find only the .hea files.
            record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
    record_names = [r for r in record_names if r not in ['100', '74', '30', '113']]

    ######### create data table file #########
    data_table_path = '/home/jeries.saleh/SSSD-ECG/src/df_data_new.csv'
    if os.path.exists(data_table_path):
        df_data = pd.read_csv(data_table_path, dtype={'file_number': str})  # keep the column as strings for consistency
        print('Data table loaded!')
    else:
        print("Preprocess again")

    ecg_ds = LongTermDataset(df_data)

    batch_size = 1
    ratio = 0.8

    length = len(record_names)
    N_test = int(round(length * ratio))

    train_recordings = record_names[:N_test]
    test_recordings = record_names[N_test:]

    train_indices = df_data[df_data['file_number'].isin(train_recordings)].index
    train_indices = [int(ind) for ind in train_indices]  # fix integer type
    # test_indices = df_data[df_data['file_number'].isin(test_recordings)].index
    # test_indices = [int(ind) for ind in test_indices]

    dl_train = DataLoader(ecg_ds, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    # dl_test = DataLoader(ecg_ds, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    # training
    n_iter = ckpt_iter + 1

    while n_iter < n_iters + 1:
        for ts, label in dl_train:
            # audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            label = label.unsqueeze(1)

            # kernel = torch.Tensor([[1, 1]])
            #
            # ts = torchaudio.functional.convolve(ts.float(), kernel)[:, 1:]
            #
            # ts = ts - torch.mean(ts)

            ts = ts.unsqueeze(1).float().cuda()

            # back-propagation
            optimizer.zero_grad()

            X = ts, label

            loss = training_loss_label(net, nn.MSELoss(), X, diffusion_hyperparams)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            iters_per_ckpt = 20000  # save checkpoint in the last iteration
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSD_ECG.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters
    train_config["output_directory"] = "/mnt/qnap/jeries/models/"

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    train(**train_config)
