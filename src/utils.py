import os
import re
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from datasets import prepare_data_incart, load_dataset, channel_stoi_default
from pathlib import Path
from tqdm.auto import tqdm


class ECGTransform(object):
    """
    ToTensor + Transforms.
    """

    def __call__(self, signal):
        # ------Your code------#
        # Transform the data type from double (float64) to single (float32) to match the later network weights.
        t_signal = signal.astype(np.single)
        # We transpose the signal to later use the lead dim as the channel... (C,L).
        # if t_signal.shape != 2:
        #     t_signal = torch.Tensor(t_signal).unsqueeze(1)
        t_signal = torch.transpose(torch.tensor(t_signal), 0, 1)
        t_signal = torch.nn.functional.normalize(t_signal)
        # ------^^^^^^^^^------#
        return t_signal  # Make sure I am a PyTorch Tensor

### evaluate stuff ###

# reference: https://github.com/helme/ecg_ptbxl_benchmarking/

def generate_results(idxs, y_true, y_pred, thresholds):
    return evaluate_experiment(y_true[idxs], y_pred[idxs], thresholds)

def evaluate_experiment(y_true, y_pred, thresholds=None):
    results = {}

    if not thresholds is None:
        # binary predictions
        y_pred_binary = apply_thresholds(y_pred, thresholds)
        # PhysioNet/CinC Challenges metrics
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']
        results['G_beta_macro'] = challenge_scores['G_beta_macro']

    # label based metric
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')

    df_result = pd.DataFrame(results, index=[0])
    return df_result

def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=True):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:,classi], y_pred[:,classi]
        TP, FP, TN, FN = 0.,0.,0.,0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i]==y_predi[i]==1:
                TP += 1./sample_weight
            if ((y_predi[i]==1) and (y_truei[i]!=y_predi[i])):
                FP += 1./sample_weight
            if y_truei[i]==y_predi[i]==0:
                TN += 1./sample_weight
            if ((y_predi[i]==0) and (y_truei[i]!=y_predi[i])):
                FN += 1./sample_weight
        f_beta_i = ((1+beta1**2)*TP)/((1+beta1**2)*TP + FP + (beta1**2)*FN)
        g_beta_i = (TP)/(TP+FP+beta2*FN)

        f_beta += f_beta_i
        g_beta += g_beta_i

    return {'F_beta_macro':f_beta/y_true.shape[1], 'G_beta_macro':g_beta/y_true.shape[1]}

def get_appropriate_bootstrap_samples(y_true, n_bootstraping_samples):
    samples=[]
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstraping_samples:
                break
    return samples

def find_optimal_cutoff_threshold(target, predicted):
    """
    Find the optimal probability cutoff point for a classification model related to event rate
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold

def find_optimal_cutoff_thresholds(y_true, y_pred):
    return [find_optimal_cutoff_threshold(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])]

def find_optimal_cutoff_threshold_for_Gbeta(target, predicted, n_thresholds=100):
    thresholds = np.linspace(0.00,1,n_thresholds)
    scores = [challenge_metrics(target, predicted>t, single=True)['G_beta_macro'] for t in thresholds]
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]

def find_optimal_cutoff_thresholds_for_Gbeta(y_true, y_pred):
    print("optimize thresholds with respect to G_beta")
    return [find_optimal_cutoff_threshold_for_Gbeta(y_true[:,k][:,np.newaxis], y_pred[:,k][:,np.newaxis]) for k in tqdm(range(y_true.shape[1]))]

def apply_thresholds(preds, thresholds):
    """
        apply class-wise thresholds to prediction score in order to get binary format.
        BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
    """
    tmp = []
    for p in preds:
        tmp_p = (p > thresholds).astype(int)
        if np.sum(tmp_p) == 0:
            tmp_p[np.argmax(p)] = 1
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp


def forward_epoch(model, dl, loss_function, optimizer, train_mode=True, desc=None, device=torch.device('cpu')):
    total_loss = 0

    with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
        for i_batch, (X, y) in enumerate(dl):

            X = X.to(device)
            y = y.to(device)

            # Forward:
            y_pred = model(X)

            # Loss:
            y_true = y.type(torch.float32)
            loss = loss_function(y_pred, y_true)
            total_loss += loss.item()

            if train_mode:  # Don't enter the gradient updating steps if not training.
                # Backward:
                optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
                loss.backward()  # get gradients

                # Optimization step:
                optimizer.step()  # use gradients

            # Store y vectors over the epoch:
            if i_batch == 0:
                y_true_epoch = y_true
                y_pred_epoch = y_pred
            else:
                y_true_epoch = torch.concat((y_true_epoch, y_true))
                y_pred_epoch = torch.concat((y_pred_epoch, y_pred))

            # Progress bar:
            pbar.update(1)

    return total_loss, y_true_epoch, y_pred_epoch


def upper_identity_matrix(m, n):
    assert n % 2 == 0, "Number of columns (n) must be even"
    matrix = torch.zeros(m, n)
    identity = torch.eye(m)
    matrix[:, :m] = identity
    return matrix.cuda()


def generate_four_leads(tensor):
    """
    return: leads12 (400,12,1000) array with the remaining 4 leads -> 12 leads in total
    """

    leadI = tensor[:, 0, :].unsqueeze(1)
    leadschest = tensor[:, 1:7, :]
    leadavf = tensor[:, 7, :].unsqueeze(1)

    leadII = (0.5 * leadI) + leadavf

    leadIII = -(0.5 * leadI) + leadavf
    leadavr = -(0.75 * leadI) - (0.5 * leadavf)
    leadavl = (0.75 * leadI) - (0.5 * leadavf)

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)

    return leads12


def save_samples(observed, target, ckpt_path, n):
    """
    logging {observed} and {target} samples in ckpt_path
    """
    if observed.shape[-1] == 1:
        return
    observed_fn = "observed_" + str(n) + ".npy"
    target_fn = "target_" + str(n) + ".npy"
    np.save(os.path.join(ckpt_path, observed_fn), observed.numpy())
    np.save(os.path.join(ckpt_path, target_fn), target.numpy())
    print(f'saved samples {observed_fn} and {target_fn}')


def define_samples(task, ds_test):
    """
    Define observed, target and cond for {task} from {ds_test};
    """

    sample = 9
    cond = ds_test[sample][1]
    # ds_test = torch.utils.data.DataLoader(ds_test, batch_size=400)

    if task == "11_lead":
        print("Now defining the observed and target for generation of 11 ecgs task")
        observed = ds_test[sample][0][0:1]  # Observed
        target = ds_test[sample][0][1:]  # Target [1,12,pred_len]
    elif task == "forecast":
        pred_len = 500  # osberved_len
        print(f"Defining observed and target for forecasting task on {1000 - pred_len} points")
        observed = ds_test[sample][0][:, :pred_len]  # Observed
        target = ds_test[sample][0][:, pred_len:]  # Target [1,12,pred_len]
        # for data, label in ds_test:
        #     observed = data[..., :pred_len]  # Observed
        #     target = data[..., pred_len:]  # Target [1,12,pred_len]
        #     cond = label
        #     break
    elif task == "incart":
        pred_len = 1000
        idx = 90001 + int(torch.rand(1) * 10000)
        print(f"Define observed and target for incart task")
        observed = ds_test[sample][0][:, idx:idx + pred_len]  # Observed
        target = ds_test[sample][0][:, idx + pred_len:idx + 2 * pred_len]  # Target [1,12,pred_len]
    else:
        observed = torch.zeros((12, 1))
        target = torch.zeros((12, 1))

    if len(observed.shape) == 2:
        observed = torch.unsqueeze(observed, 0)
        target = torch.unsqueeze(target, 0)
        cond = torch.unsqueeze(cond, 0)

    return observed, target, cond


def extract_info_from_file(filename):
    """
    extract age, and diagnosis from .hea file
    """
    age_pattern = r'#<age>:\s*(\d+)'
    age = None
    diagnosis = 'None'
    with open(filename, 'r') as file:
        for line in file:
            age_match = re.match(age_pattern, line)
            if age_match:
                age = int(age_match.group(1))
                match = re.search(r'<diagnoses>\s*(.*)', line)
                if match:
                    diagnosis = match.group(1)
    return age, diagnosis.split(', ')


def process_files_in_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".hea"):
            filepath = os.path.join(directory, filename)
            age, diagnosis = extract_info_from_file(filepath)
            if age is not None:
                data.append({'filename': filename[:3], 'age': age, 'diagnosis': diagnosis})
    return pd.DataFrame(data)


def dataset_split_incart(data_dir, target_dir, recreate_data=True):
    print(f'Working on {target_dir} folder to split data for train, val and test')
    df_labels = process_files_in_directory(data_dir)

    # Prepare the dataset (INCART)
    prepare_data_incart(data_dir, df_labels, channels=12, channel_stoi=channel_stoi_default,
                        target_folder=target_dir, recreate_data=recreate_data)

    df_mapped = load_dataset(target_dir, df_mapped=False, incart=True)

    df_train = df_mapped[df_mapped.strat_fold == 1]
    df_val = df_mapped[df_mapped.strat_fold == 2]
    df_test = df_mapped[df_mapped.strat_fold == 3]

    return df_train, df_val, df_test


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


def npys_to_memmap_batched(npys, target_filename, max_len=0, delete_npys=True, batch_length=900000):
    memmap = None
    start = np.array(
        [0])  # start_idx in current memmap file (always already the next start- delete last token in the end)
    length = []  # length of segment
    filenames = []  # memmap files
    file_idx = []  # corresponding memmap file for sample
    shape = []  # shapes of all memmap files

    data = []
    data_lengths = []
    dtype = None

    for idx, npy in tqdm(list(enumerate(npys))):

        data.append(np.load(npy, allow_pickle=True))
        data_lengths.append(len(data[-1]))

        if (idx == len(npys) - 1 or np.sum(data_lengths) > batch_length):  # flush
            data = np.concatenate(data)
            if (memmap is None or (max_len > 0 and start[-1] > max_len)):  # new memmap file has to be created
                if (max_len > 0):
                    filenames.append(
                        target_filename.parent / (target_filename.stem + "_" + str(len(filenames)) + ".npy"))
                else:
                    filenames.append(target_filename)

                shape.append([np.sum(data_lengths)] + [l for l in data.shape[1:]])  # insert present shape

                if (memmap is not None):  # an existing memmap exceeded max_len
                    del memmap
                # create new memmap
                start[-1] = 0
                start = np.concatenate([start, np.cumsum(data_lengths)])
                length = np.concatenate([length, data_lengths])

                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
            else:
                # append to existing memmap
                start = np.concatenate([start, start[-1] + np.cumsum(data_lengths)])
                length = np.concatenate([length, data_lengths])
                shape[-1] = [start[-1]] + [l for l in data.shape[1:]]
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple(shape[-1]))

            # store mapping memmap_id to memmap_file_id
            file_idx = np.concatenate([file_idx, [(len(filenames) - 1)] * len(data_lengths)])
            # insert the actual data
            memmap[start[-len(data_lengths) - 1]:start[-len(data_lengths) - 1] + len(data)] = data[:]
            memmap.flush()
            dtype = data.dtype
            data = []  # reset data storage
            data_lengths = []

    start = start[:-1]  # remove the last element
    # cleanup
    for npy in npys:
        if (delete_npys is True):
            npy.unlink()
    del memmap

    # convert everything to relative paths
    filenames = [f.name for f in filenames]
    # save metadata
    np.savez(target_filename.parent / (target_filename.stem + "_meta.npz"), start=start, length=length, shape=shape,
             file_idx=file_idx, dtype=dtype, filenames=filenames)


def npys_to_memmap(npys, target_filename, max_len=0, delete_npys=True):
    memmap = None
    start = []  # start_idx in current memmap file
    length = []  # length of segment
    filenames = []  # memmap files
    file_idx = []  # corresponding memmap file for sample
    shape = []

    for _, npy in tqdm(list(enumerate(npys))):
        data = np.load(npy, allow_pickle=True)
        if (memmap is None or (max_len > 0 and start[-1] + length[-1] > max_len)):
            if (max_len > 0):
                filenames.append(target_filename.parent / (target_filename.stem + "_" + str(len(filenames)) + ".npy"))
            else:
                filenames.append(target_filename)

            if (memmap is not None):  # an existing memmap exceeded max_len
                shape.append([start[-1] + length[-1]] + [l for l in data.shape[1:]])
                del memmap
            # create new memmap
            start.append(0)
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
        else:
            # append to existing memmap
            start.append(start[-1] + length[-1])
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+',
                               shape=tuple([start[-1] + length[-1]] + [l for l in data.shape[1:]]))

        # store mapping memmap_id to memmap_file_id
        file_idx.append(len(filenames) - 1)
        # insert the actual data
        memmap[start[-1]:start[-1] + length[-1]] = data[:]
        memmap.flush()
        if (delete_npys is True):
            npy.unlink()
    del memmap

    # append final shape if necessary
    if (len(shape) < len(filenames)):
        shape.append([start[-1] + length[-1]] + [l for l in data.shape[1:]])
    # convert everything to relative paths
    filenames = [f.name for f in filenames]
    # save metadata
    np.savez(target_filename.parent / (target_filename.stem + "_meta.npz"), start=start, length=length, shape=shape,
             file_idx=file_idx, dtype=data.dtype, filenames=filenames)


def reformat_as_memmap(df, target_filename, data_folder=None, annotation=False, max_len=0, delete_npys=True,
                       col_data="data", col_label="label", batch_length=0):
    target_filename = Path(target_filename)
    data_folder = Path(data_folder)

    npys_data = []
    npys_label = []

    for _, row in df.iterrows():
        npys_data.append(data_folder / row[col_data] if data_folder is not None else row[col_data])
        if (annotation):
            npys_label.append(data_folder / row[col_label] if data_folder is not None else row[col_label])
    if (batch_length == 0):
        npys_to_memmap(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys)
    else:
        npys_to_memmap_batched(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys,
                               batch_length=batch_length)
    if (annotation):
        if (batch_length == 0):
            npys_to_memmap(npys_label, target_filename.parent / (target_filename.stem + "_label.npy"), max_len=max_len,
                           delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(npys_label, target_filename.parent / (target_filename.stem + "_label.npy"),
                                   max_len=max_len, delete_npys=delete_npys, batch_length=batch_length)

    # replace data(filename) by integer
    df_mapped = df.copy()
    df_mapped[col_data + "_original"] = df_mapped.data
    df_mapped[col_data] = np.arange(len(df_mapped))

    df_mapped.to_pickle(target_filename.parent / ("df_" + target_filename.stem + ".pkl"))
    return df_mapped


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def get_windows(input_length=1000, window_size=100, overlap=10):
    """
    A function for returning the windows to use for MultiDiffusion. Based on [1]

    References
    ----------
    [1] MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation
    """
    stride = window_size - overlap
    num_windows = (input_length - window_size) // stride + 1
    if (input_length - window_size) % stride != 0:
        num_windows += 1
    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        if end > input_length:
            print(f"last index was {end}")
            raise Exception("Choose different overlap or window_size")
        windows.append((start, end))

    return windows


def sampling_label(net, size, diffusion_hyperparams, cond=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    cond: conditioning as integer tensor
    guidance_weight: weight for classifier-free guidance (if trained with conditioning_dropout>0)

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta

            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(
                Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}
    return x


def sampling_pgdm(net, size, diffusion_hyperparams, cond=None, observed=None):
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    B, C, L = size
    N = observed.shape[-1]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    print('begin sampling, total number of reverse steps = %s' % T)
    x = std_normal(size)  # shape observed_hat + target_hat
    c = upper_identity_matrix(N, size[-1])
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, diffusion_steps,))  # epsilon_theta

            x_hat = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(
                Alpha[t])  # x_t_hat

            z = torch.matmul(torch.t(c), (observed.reshape(B, N, C) - torch.matmul(c, x_hat.reshape(B, L, C))))

            g = (1 / 0.001) * (net((x + 0.001 * z.reshape(B, C, L), cond, diffusion_steps,)) - net(
                (x, cond, diffusion_steps,)))

            if t > 0:
                x = x + Sigma[t] * std_normal(size) + g  # add the variance term to x_{t-1}
    return x


def sampling_guidance(net, size, diffusion_hyperparams, cond=None, observed=None):
    """
    Inference using guidance from option 3 in [1]

    Parameters:
    -----------
    net (torch network):            src
    size (tuple):                   size of tensor to be generated,
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    cond:                           conditioning as integer tensor (labels)
    observed:                       observed time series

    Returns:
    --------
    the generated ECG(s) in torch.tensor, shape=size

    References
    ----------
    [1] https://www.youtube.com/watch?v=Te5kibGjsUU
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    B, C, L = size
    N = observed.shape[-1]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    # s_k = 0.001
    print('begin sampling, total number of reverse steps = %s' % T)
    x = std_normal(size)  # x_T
    c = upper_identity_matrix(N, size[-1])
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, diffusion_steps,))

            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # D(x,sigma)

            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

        temp = (torch.matmul(c, x.reshape(B, L, C)).reshape(B, C, N) - observed).reshape(B, N, C)  # C*(D(x,sigma) - y

        x = x - torch.matmul(torch.t(c), temp).reshape(B, C, L)  # denoiser - C^T * [C*(D(x,sigma) - y]

        # x[..., :N] = observed
    return x


def sampling(net, size, diffusion_hyperparams, cond=None, observed=None, guidance=False, refine=False, alpha=0.5):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)
    With option to forecast using Guided Diffusion and Refine the forecast as in [1]

    Parameters:
    -----------
    net (torch network):            src
    size (tuple):                   size of tensor to be generated,
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    cond:                           conditioning as integer tensor (labels)
    observed:                       observed time series as in [1]
    guidance (bool):                use guided diffusion
    refine (bool):                  refine the prediction as used in [1]
    alpha (float):                  alters guidance:noise ratio

    guidance_weight: weight for classifier-free guidance (if trained with conditioning_dropout>0)

    Returns:
    --------
    the generated ECG(s) in torch.tensor, shape=size

    References
    ----------
    [1] Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting. (2023)
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    g = 0
    _, __, L = observed.shape  # (B,C,L)
    N = size[-1] - L
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)
    temp = torch.zeros(size).cuda()
    for _ in range(10):
        x = std_normal(size)  # shape observed_hat + target_hat
        # target = x[:,:,L-1:]
        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
                epsilon_theta = net((x, cond, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                if guidance:

                    # y_hat = (x_{t} - sqrt(1-\alpha_bar_t) * eps_theta) / sqrt(\alpha_bar)
                    # guided diffusion of the target according to observed
                    with torch.enable_grad():
                        x.requires_grad_(True)
                        y_hat = (x - torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha_bar[t])
                        mse = F.mse_loss(y_hat, observed, reduction="sum")
                        g = -torch.autograd.grad(mse, x, grad_outputs=torch.ones_like(mse))[0]

                # update x_{t-1} to \mu_\theta(x_t)
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])

                if t > 0:
                    alpha = 0 if not guidance else alpha
                    x = x + Sigma[t] * (alpha * g + (1 - alpha) * std_normal(size))  # add the variance term to x_{t-1}

                if refine:
                    # idx = torch.tensor((L-N)*torch.rand(1), dtype=int) # [:,:,idx:idx+N]
                    x = refine_ld(
                        init=x,
                        observed=observed,
                        step_size=0.01,
                        noise_scale=0.01,
                        n_steps=2,
                        epsilon_theta=epsilon_theta
                    )

        temp = temp + x
    # x = torch.div(temp, 10)
    return x


def refine_ld(init, observed, step_size, noise_scale, n_steps, epsilon_theta):
    """
    Energy-Based refinement [1] using Langevin Dynamics

    Returns
    -------
    Updated point.

    References
    ----------
    [1] Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting. (2023)
    """
    y = init
    sqrt_2eta = torch.sqrt(torch.tensor(2 * step_size))
    for _ in range(n_steps):
        with torch.enable_grad():
            y.requires_grad_(True)
            e = F.mse_loss(y, observed, reduction="sum") \
                + F.mse_loss(std_normal(epsilon_theta.shape), epsilon_theta, reduction="sum")

            v = -torch.autograd.grad(e, y, grad_outputs=torch.ones_like(e))[0]  # grad(E_\theta) w.r.t y

        # y_(t+1) = y_t - step_size * grad(E) + sqrt(2*step_size*gamma)*noise
        y = y.detach() + step_size * v + sqrt_2eta * noise_scale * torch.randn_like(y)

    return y