import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import pandas as pd
import torch
from models.models import SequentialNet
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from sklearn.metrics import accuracy_score
from itertools import repeat
from datasets import LongTermDataset

import multiprocessing

from utils import *

synthetic = True
train_from_checkpoint = True
train = True

data_path = '/mnt/qnap/jeries/physionet.org/files/ltafdb/1.0.0/'
results_path = '/home/jeries.saleh/synth-ecg-analysis/results/'

record_names = []
for file in os.listdir(data_path):
    if file.endswith('.hea'):  # we find only the .hea files.
        record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
# print(record_names)
# print(f'Number of records: {len(record_names)}')
record_names = [r for r in record_names if r not in ['100', '74', '30', '113']]

fs = 128


###### Get the df_data ######
data_table_path = '/home/jeries.saleh/SSSD-ECG/src/df_data_new.csv'
assert os.path.exists(data_table_path) == True

batch_size = 100
# ratio = 0.3
# ratio = 0.01

df_data = pd.read_csv(data_table_path, dtype={'file_number': str})  # keep the column as strings for consistency
print('Data table loaded!')

if synthetic:
    ecg_ds_train = LongTermDataset(data_table=None, transform=None, synthetic=synthetic)
    dl_train = DataLoader(ecg_ds_train, batch_size=batch_size)
    ecg_ds = LongTermDataset(df_data, ECGTransform(), synthetic=False)
else:
    ecg_ds = LongTermDataset(df_data, ECGTransform(), synthetic=synthetic)

length = len(record_names)
N_train = int(round(length * 0.3))
N_test = int(round(length * 0.4))

# train_recordings = record_names[:int(round(length * 0.3))] ## only for calculating threshold
train_recordings = record_names[:N_train]
test_recordings = record_names[N_train:N_test]

train_indices = df_data[df_data['file_number'].isin(train_recordings)].index
train_indices = [int(ind) for ind in train_indices]  # fix integer type
test_indices = df_data[df_data['file_number'].isin(test_recordings)].index
test_indices = [int(ind) for ind in test_indices]

if synthetic:
    dl_test = DataLoader(ecg_ds, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))
else:
    dl_train = DataLoader(ecg_ds, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    dl_test = DataLoader(ecg_ds, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

print("Dataloaders are ready for training")

# pos_weight = 10.263855934143066 LIGHTER
pos_weight = 9.38
pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# with open("/home/jeries.saleh/SSSD-ECG/src/sssd/config/config_SSSD_ECG.json") as f:
#     data = f.read()
#
# config = json.loads(data)
# print(config)
#
# diffusion_config = config["diffusion_config"]  # basic hyperparameters
#
# diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters
#
# model_config = config['wavenet_config']
#
# model = SSSD_ECG(**model_config).cuda()

model = SequentialNet((batch_size, 1, 1920), fs).cuda()

if train_from_checkpoint:
    checkpoint = torch.load('/home/jeries.saleh/synth-ecg-analysis/results/synth/seq_9_october_low_data.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f'Successfully loaded saved model')

learning_rate = 0.00001

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

epochs = 300

train_loss_vec = []
test_loss_vec = []

best_accuracy = 0
best_epoch = 0

if train:
    print("Training loop incoming ...")
    for i_epoch in range(epochs):
        print(f'Epoch: {i_epoch + 1}/{epochs}')

        model.train()
        train_loss, y_true_train, y_pred_train = forward_epoch(model, dl_train, loss_function, optimizer, train_mode=True,
                                                               desc='Train', device=torch.device('cuda'))

        # Metrics:
        train_loss = train_loss / len(dl_train)
        train_loss_vec.append(train_loss)
        train_accuracy = accuracy_score(y_true_train.cpu(), (
                y_pred_train.cpu().detach() > 0.5) * 1)

        # test_loss = test_loss / len(dl_test)
        # test_loss_vec.append(test_loss)
        # test_accuracy = accuracy_score(y_true_test.cpu(), (y_pred_test.cpu().detach() > 0.5) * 1)

        # print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
        #       test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')
        print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)}')

        ### save every 20 epoch
        if  i_epoch%50==0: # 50 if synthetic
            torch.save(model.state_dict(), results_path + '/synth/seq_9_october_low_data_with_synthetic' + str(i_epoch) + '.pt')
            best_epoch = i_epoch

        if i_epoch - best_epoch > 30:
            print('Early stopping triggered')
            break

    # print(f'best test accuracy was {best_accuracy=} and the best epoch was {best_epoch=}')

    # np.save(os.path.join(results_path, 'train_loss_vec.npy'), np.array(train_loss_vec))
    # np.save(os.path.join(results_path, 'test_loss_vec.npy'), np.array(test_loss_vec))
    print("Evaluation mode ...")

    model.eval()
    with torch.no_grad():
        test_loss, y_true_test, y_pred_test = forward_epoch(model, dl_test, loss_function, optimizer,
                                                            train_mode=False,
                                                            desc='Test', device=torch.device('cuda'))

    np.save(os.path.join(results_path, 'y_true_synth_ld.npy'), y_true_test.detach().cpu().numpy())
    np.save(os.path.join(results_path, 'y_pred_synth_ld.npy'), y_pred_test.detach().cpu().numpy())

    y_pred_train = np.where(y_pred_test > 0, 1, 0)
    y_pred_test = np.where(y_pred_test > 0, 1, 0)

    te_df_point = generate_results(range(len(y_true_test)), y_true_test[..., np.newaxis],
                                   y_pred_test[..., np.newaxis], thresholds=0.0)

else:
    print("Evaluation mode ...")

    checkpoint = torch.load('/home/jeries.saleh/synth-ecg-analysis/results/synth/seq_9_october_synthetic_lr_high_100.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f'Successfully loaded saved model')

    model.eval()
    with torch.no_grad():
        test_loss, y_true_test, y_pred_test = forward_epoch(model, dl_test, loss_function, optimizer,
                                                            train_mode=False,
                                                            desc='Test', device=torch.device('cuda'))

    np.save(os.path.join(results_path, 'y_true_synth_3.npy'), y_true_test.detach().cpu().numpy())
    np.save(os.path.join(results_path, 'y_pred_synth_3.npy'), y_pred_test.detach().cpu().numpy())

    y_pred_test = np.load(results_path + "y_pred_synth_3.npy")
    y_true_test = np.load(results_path + "y_true_synth_3.npy")

    y_pred_train = np.where(y_pred_test > 0, 1, 0)
    y_pred_test = np.where(y_pred_test > 0, 1, 0)


    te_df_point = generate_results(range(len(y_true_test)), y_true_test[..., np.newaxis],
                                   y_pred_test[..., np.newaxis], thresholds=0.0)

    print(te_df_point)


# te_df = pd.concat(pool.starmap(
#     generate_results,
#     zip(test_samples, repeat(y_true_test), repeat(y_pred_test), repeat(thresholds))))
# te_df_result = pd.DataFrame(
#     np.array([
#     te_df_point.mean().values,
#     te_df.mean().values,
#     te_df.quantile(0.05).values,
#     te_df.quantile(0.95).values]),
# columns=te_df.columns,
# index=['point', 'mean', 'lower', 'upper'])
