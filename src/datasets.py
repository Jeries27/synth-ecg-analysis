import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import random
from tqdm.auto import tqdm
import resampy
import wfdb


try:
    import pickle5 as pickle
except ImportError as e:
    import pickle

channel_stoi_default = {"i": 0, "ii": 1, "v1": 2, "v2": 3, "v3": 4, "v4": 5, "v5": 6, "v6": 7, "iii": 8, "avr": 9,
                        "avl": 10, "avf": 11, "vx": 12, "vy": 13, "vz": 14}

# Define the diagnoses dictionary
diagnoses_dict = {
    'Coronary artery disease': 0,
    'arterial hypertension': 1,
    'Acute MI': 2,
    'Earlier MI': 3,
    'Transient ischemic attack': 4,
    'Left ventricular hypertrophy': 5,
    'Sinus node dysfunction': 6,
    'None': 7
}

diagnoses_dict_incart_to_ptb = {
    'Coronary artery disease': 2,  # ST/T Change 2
    'arterial hypertension': 2,  # Can it be in ST/T change?
    'Acute MI': 1,  # Myocardial Infarction 1
    'Earlier MI': 1,  # Myocardial Infarction 1
    'Transient ischemic attack': 2,  # ST/T Change 2
    'left ventricular hypertrophy': 4,  # Hypertrophy 4
    'Sinus node dysfunction': 3,  # Conduction Disturbance 3
    'None': 0  # NORM 0
}

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from collections import namedtuple

tsdata_static = namedtuple("tsdata_static", ("data", "label", "static"))
tsdata = namedtuple("tsdata", ("data", "label"))

import pathlib

pathlib.WindowsPath = pathlib.PosixPath

def save_dataset(df, target_root, lbl_itos=None, mean=None, std=None, filename_postfix="", protocol=4, incart=False):
    target_root = Path(target_root)
    df.to_pickle(target_root / ("df" + filename_postfix + ".pkl"), protocol=protocol)
    if incart:
        return
    if (isinstance(lbl_itos, dict)):  # dict as pickle
        outfile = open(target_root / ("lbl_itos" + filename_postfix + ".pkl"), "wb")
        pickle.dump(lbl_itos, outfile, protocol=protocol)
        outfile.close()
    else:  # array
        np.save(target_root / ("lbl_itos" + filename_postfix + ".npy"), lbl_itos)

    np.save(target_root / ("mean" + filename_postfix + ".npy"), mean)
    np.save(target_root / ("std" + filename_postfix + ".npy"), std)


def load_dataset(target_root, filename_postfix="", df_mapped=True, incart=False):
    target_root = Path(target_root)

    if df_mapped:
        df = pickle.load(open(target_root / ("df_memmap" + filename_postfix + ".pkl"), "rb"))
    else:
        df = pickle.load(open(target_root / ("df" + filename_postfix + ".pkl"), "rb"))
    if incart:
        return df
    if (target_root / ("lbl_itos" + filename_postfix + ".pkl")).exists():  # dict as pickle
        infile = open(target_root / ("lbl_itos" + filename_postfix + ".pkl"), "rb")
        lbl_itos = pickle.load(infile)
        infile.close()
    else:  # array
        lbl_itos = np.load(target_root / ("lbl_itos" + filename_postfix + ".npy"))

    mean = np.load(target_root / ("mean" + filename_postfix + ".npy"))
    std = np.load(target_root / ("std" + filename_postfix + ".npy"))
    return df, lbl_itos, mean, std


# Cell

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=12, channel_stoi=None):
    # ,skimage_transform=True,interpolation_order=3):
    channel_labels = [c.lower() for c in channel_labels]
    # https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs / fs
    timesteps_new = int(len(sigbufs) * factor)
    if (channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i, cl in enumerate(channel_labels):
            if (cl in channel_stoi.keys() and channel_stoi[cl] < channels):
                data[:, channel_stoi[cl]] = resampy.resample(sigbufs[:, i], fs, target_fs).astype(np.float32)
    else:
        data = resampy.resample(sigbufs, fs, target_fs, axis=0).astype(np.float32)
    return data


def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col + "_length"] = df[col].apply(
        lambda x: len(np.load(x if data_folder is None else data_folder / x, allow_pickle=True)))


def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col + "_mean"] = df[col].apply(
        lambda x: np.mean(np.load(x if data_folder is None else data_folder / x, allow_pickle=True), axis=axis))


def dataset_add_median_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with median'''
    df[col + "_median"] = df[col].apply(
        lambda x: np.median(np.load(x if data_folder is None else data_folder / x, allow_pickle=True), axis=axis))


def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col + "_std"] = df[col].apply(
        lambda x: np.std(np.load(x if data_folder is None else data_folder / x, allow_pickle=True), axis=axis))


def dataset_get_stats(df, col="data", simple=True):
    '''creates (weighted) means and stds from mean, std and length cols of the df'''
    if (simple):
        return df[col + "_mean"].mean(), df[col + "_std"].mean()
    else:
        # https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        # or https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
        def combine_two_means_vars(x1, x2):
            (mean1, var1, n1) = x1
            (mean2, var2, n2) = x2
            mean = mean1 * n1 / (n1 + n2) + mean2 * n2 / (n1 + n2)
            var = var1 * n1 / (n1 + n2) + var2 * n2 / (n1 + n2) + n1 * n2 / (n1 + n2) / (n1 + n2) * np.power(
                mean1 - mean2, 2)
            return (mean, var, (n1 + n2))

        def combine_all_means_vars(means, vars, lengths):
            inputs = list(zip(means, vars, lengths))
            result = inputs[0]

            for inputs2 in inputs[1:]:
                result = combine_two_means_vars(result, inputs2)
            return result

        means = list(df[col + "_mean"])
        vars = np.power(list(df[col + "_std"]), 2)
        lengths = list(df[col + "_length"])
        mean, var, length = combine_all_means_vars(means, vars, lengths)
        return mean, np.sqrt(var)


def convert_to_multihot(diagnoses):
    multihot_encoding = [0] * (max(diagnoses_dict_incart_to_ptb.values()) + 1)
    for diagnosis in diagnoses:
        if diagnosis in diagnoses_dict_incart_to_ptb:
            multihot_encoding[diagnoses_dict_incart_to_ptb[diagnosis]] = 1
    return multihot_encoding


def prepare_data_incart(data_path, df_labels, target_fs=100, channels=12, channel_stoi=channel_stoi_default,
                        target_folder=None,
                        recreate_data=True):
    target_root_incart = Path(".") if target_folder is None else target_folder  # mnt/qnap/jeries/incart
    target_root_incart.mkdir(parents=True, exist_ok=True)

    if recreate_data:
        records = target_root_incart / "incart_database.csv"
        df_records = pd.read_csv(records, index_col="ecg_id")
        df_records = pd.merge(df_records, df_labels, on='filename', how='left')

        df_records['multihot_encoding'] = df_records['diagnosis'].apply(convert_to_multihot)

        filenames_npy = []
        filenames_int = []
        filenames_data = []
        for id, row in tqdm(list(df_records.iterrows())):
            filename = data_path / row["filename"]
            sigbufs, header = wfdb.rdsamp(str(filename))
            data = resample_data(sigbufs=sigbufs, channel_stoi=channel_stoi, channel_labels=header['sig_name'],
                                 fs=header['fs'], target_fs=target_fs,  # header['fs']
                                 channels=channels)
            np.save(target_root_incart / (filename.stem + ".npy"), data)
            filenames_npy.append(Path(filename.stem + ".npy"))  ##
            filenames_int.append(np.int32(filename.stem[1:]))  ##
            filenames_data.append(Path(filename.stem + ".dat"))
        df_records["data"] = filenames_int
        df_records["filenames_dat"] = filenames_data
        df_records["filenames_npy"] = filenames_npy

        # save
        save_dataset(df=df_records, target_root=target_root_incart, incart=True)
    else:
        df_records = load_dataset(target_root_incart, df_mapped=False, incart=True)
    return df_records


# Cell
def prepare_data_ptb_xl(data_path, min_cnt=10, target_fs=100, channels=12, channel_stoi=channel_stoi_default,
                        target_folder=None, recreate_data=True):
    target_root_ptb_xl = Path(".") if target_folder is None else target_folder
    # print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if recreate_data:
        # reading df
        ptb_xl_csv = data_path / "ptbxl_database.csv"
        df_ptb_xl = pd.read_csv(ptb_xl_csv, index_col="ecg_id")
        # print(df_ptb_xl.columns)
        df_ptb_xl.scp_codes = df_ptb_xl.scp_codes.apply(lambda x: eval(x.replace("nan", "np.nan")))

        # preparing labels
        ptb_xl_label_df = pd.read_csv(data_path / "scp_statements.csv")
        ptb_xl_label_df = ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

        ptb_xl_label_diag = ptb_xl_label_df[ptb_xl_label_df.diagnostic > 0]
        ptb_xl_label_form = ptb_xl_label_df[ptb_xl_label_df.form > 0]
        ptb_xl_label_rhythm = ptb_xl_label_df[ptb_xl_label_df.rhythm > 0]

        diag_class_mapping = {}
        diag_subclass_mapping = {}
        for id, row in ptb_xl_label_diag.iterrows():
            if isinstance(row["diagnostic_class"], str):
                diag_class_mapping[id] = row["diagnostic_class"]
            if isinstance(row["diagnostic_subclass"], str):
                diag_subclass_mapping[id] = row["diagnostic_subclass"]

        df_ptb_xl["label_all"] = df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()])
        df_ptb_xl["label_diag"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
        df_ptb_xl["label_form"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
        df_ptb_xl["label_rhythm"] = df_ptb_xl.scp_codes.apply(
            lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

        df_ptb_xl["label_diag_subclass"] = df_ptb_xl.label_diag.apply(
            lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
        df_ptb_xl["label_diag_superclass"] = df_ptb_xl.label_diag.apply(
            lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])

        df_ptb_xl["dataset"] = "ptb_xl"
        # filter and map (can be reapplied at any time)
        df_ptb_xl, lbl_itos_ptb_xl = map_and_filter_labels(df_ptb_xl, min_cnt=min_cnt,
                                                           lbl_cols=["label_all", "label_diag", "label_form",
                                                                     "label_rhythm", "label_diag_subclass",
                                                                     "label_diag_superclass"])

        filenames = []
        for id, row in tqdm(list(df_ptb_xl.iterrows())):
            # always start from 500Hz and sample down
            filename = data_path / row[
                "filename_hr"]  # data_path/row["filename_lr"] if target_fs<=100 else data_path/row["filename_hr"]
            sigbufs, header = wfdb.rdsamp(str(filename))
            data = resample_data(sigbufs=sigbufs, channel_stoi=channel_stoi, channel_labels=header['sig_name'],
                                 fs=header['fs'], target_fs=target_fs, channels=channels)
            assert (target_fs <= header['fs'])
            np.save(target_root_ptb_xl / (filename.stem + ".npy"), data)
            filenames.append(Path(filename.stem + ".npy"))
        df_ptb_xl["data"] = filenames

        # add means and std
        dataset_add_mean_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl, data_folder=target_root_ptb_xl)
        # dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        # dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        # save means and stds
        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        # save
        save_dataset(df=df_ptb_xl, lbl_itos=lbl_itos_ptb_xl, mean=mean_ptb_xl, std=std_ptb_xl,
                     target_root=target_root_ptb_xl)
    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl, df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl


def map_and_filter_labels(df, min_cnt, lbl_cols):
    # filter labels
    def select_labels(labels, min_cnt=10):
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt >= min_cnt)[0]])

    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl = {}
    for selection in lbl_cols:
        if (min_cnt > 0):
            label_selected = select_labels(df_ptb_xl[selection], min_cnt=min_cnt)
            df_ptb_xl[selection + "_filtered"] = df_ptb_xl[selection].apply(
                lambda x: [y for y in x if y in label_selected])
            lbl_itos_ptb_xl[selection + "_filtered"] = np.array(
                sorted(list(set([x for sublist in df_ptb_xl[selection + "_filtered"] for x in sublist]))))
            lbl_stoi = {s: i for i, s in enumerate(lbl_itos_ptb_xl[selection + "_filtered"])}
            df_ptb_xl[selection + "_filtered_numeric"] = df_ptb_xl[selection + "_filtered"].apply(
                lambda x: [lbl_stoi[y] for y in x])
        # also lbl_itos and ..._numeric col for original label column
        lbl_itos_ptb_xl[selection] = np.array(
            sorted(list(set([x for sublist in df_ptb_xl[selection] for x in sublist]))))
        lbl_stoi = {s: i for i, s in enumerate(lbl_itos_ptb_xl[selection])}
        df_ptb_xl[selection + "_numeric"] = df_ptb_xl[selection].apply(lambda x: [lbl_stoi[y] for y in x])
    return df_ptb_xl, lbl_itos_ptb_xl


def multihot_encode(x, num_classes):
    res = np.zeros(num_classes, dtype=np.float32)
    for y in x:
        res[y] = 1
    return res


def load_data(
        data_dir,
        split,
        label="label_all",
):
    """
    For a dataset, create a DataFrame variable for dataloader

    Parameters
    ----------
    data_dir (Path):    a dataset directory.
    split (str):          train / val / test
    label (str):        labels from dataset to use
    """
    # Prepare the dataset
    # df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_data_incart(data_folder_incart, min_cnt=0,
    #                                                                           channels=12,
    #                                                                           channel_stoi=channel_stoi_default,
    #                                                                           target_folder=target_folder_ptb_xl)
    if not data_dir:
        raise ValueError("unspecified data directory")

    df_mapped, lbl_itos, mean, std = load_dataset(data_dir)
    lbl_itos = np.array(lbl_itos[label])
    df_mapped["label"] = df_mapped[label + "_numeric"].apply(lambda x: multihot_encode(x, len(lbl_itos)))
    max_fold_id = df_mapped.strat_fold.max()

    if split == "train":
        return df_mapped[df_mapped.strat_fold < max_fold_id - 1]
    elif split == "test":
        return df_mapped[df_mapped.strat_fold == max_fold_id]
    else:
        raise ValueError("unspecified train / test")


class TimeseriesDatasetCrops(Dataset):
    """timeseries dataset with partial crops."""

    def __init__(
            self,
            df,
            output_size,
            chunk_length,
            min_chunk_length,
            memmap_filename=None,
            npy_data=None,
            random_crop=True,
            data_folder=None,
            num_classes=2,
            copies=0,
            col_lbl="label",
            cols_static=None,
            stride=None,
            start_idx=0,
            annotation=False,
            transforms=None,
            sample_items_per_record=1,
            incart=False
    ):
        """
        accepts three kinds of input:
        1) filenames pointing to aligned numpy arrays [timesteps,channels,...] for data and either integer labels or filename pointing to numpy arrays[timesteps,...] e.g. for annotations
        2) memmap_filename to memmap file (same argument that was passed to reformat_as_memmap) for data [concatenated,...] and labels- data column in df corresponds to index in this memmap
        3) npy_data [samples,ts,...] (either path or np.array directly- also supporting variable length input) - data column in df corresponds to sampleid

        transforms: list of callables (transformations) or (preferred) single instance e.g. from torchvision.transforms.Compose (applied in the specified order i.e. leftmost element first)

        col_lbl = None: return dummy label 0 (e.g. for unsupervised pretraining)
        cols_static: (optional) list of cols with extra static information
        """
        assert not ((memmap_filename is not None) and (npy_data is not None))  # data is not empty
        # require integer entries if using memmap or npy
        # keys (in column data) have to be unique
        assert (len(df["data"].unique()) == len(df))

        self.timeseries_df_data = np.array(df["data"])
        if self.timeseries_df_data.dtype not in [np.int16, np.int32, np.int64]:
            assert (memmap_filename is None and npy_data is None)  # only for filenames in mode files
            self.timeseries_df_data = np.array(df["data"].astype(str)).astype(np.string_)

        if isinstance(df[col_lbl].iloc[0], list) or isinstance(df[col_lbl].iloc[0],
                                                               np.ndarray):  # stack arrays/lists for proper batching
            self.timeseries_df_label = np.stack(df[col_lbl])
        else:  # single integers/floats
            self.timeseries_df_label = np.array(df[col_lbl])

        if self.timeseries_df_label.dtype not in [np.int16, np.int32, np.int64, np.float32,
                                                  np.float64]:  # everything else cannot be batched anyway mp.Manager().list(self.timeseries_df_label)
            assert (annotation and memmap_filename is None and npy_data is None)  # only for filenames in mode files
            self.timeseries_df_label = np.array(df[col_lbl].apply(lambda x: str(x))).astype(np.string_)

        if cols_static is not None:
            self.timeseries_df_static = np.array(df[cols_static]).astype(np.float32)
            self.static = True
        else:
            self.static = False

        self.output_size = output_size
        self.data_folder = data_folder
        self.transforms = transforms
        if isinstance(self.transforms, list) or isinstance(self.transforms, np.ndarray):
            print("Warning: the use of lists as arguments for transforms is discouraged")
        self.annotation = annotation
        self.col_lbl = col_lbl

        self.c = num_classes

        self.mode = "files"

        if memmap_filename is not None:
            self.memmap_meta_filename = memmap_filename.parent / (memmap_filename.stem + "_meta.npz")
            self.mode = "memmap"
            memmap_meta = np.load(self.memmap_meta_filename, allow_pickle=True)
            self.memmap_start = memmap_meta["start"]
            self.memmap_shape = memmap_meta["shape"]
            self.memmap_length = memmap_meta["length"]
            self.memmap_file_idx = memmap_meta["file_idx"]
            self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
            self.memmap_filenames = np.array(memmap_meta["filenames"]).astype(
                np.string_)  # save as byte to avoid issue with mp
            if annotation:
                memmap_meta_label = np.load(self.memmap_meta_filename.parent / (
                        "_".join(self.memmap_meta_filename.stem.split("_")[:-1]) + "_label_meta.npz"),
                                            allow_pickle=True)
                self.memmap_shape_label = memmap_meta_label["shape"]
                self.memmap_filenames_label = np.array(memmap_meta_label["filenames"]).astype(np.string_)
                self.memmap_dtype_label = np.dtype(str(memmap_meta_label["dtype"]))
        elif npy_data is not None:
            self.mode = "npy"
            if isinstance(npy_data, np.ndarray) or isinstance(npy_data, list):
                self.npy_data = np.array(npy_data)
                assert (annotation is False)
            else:
                self.npy_data = np.load(npy_data, allow_pickle=True)
            if (annotation):
                self.npy_data_label = np.load(npy_data.parent / (npy_data.stem + "_label.npy"), allow_pickle=True)

        self.random_crop = random_crop
        self.sample_items_per_record = sample_items_per_record
        self.incart = incart

        self.df_idx_mapping = []
        self.start_idx_mapping = []
        self.end_idx_mapping = []

        for df_idx, (id, row) in enumerate(df.iterrows()):
            if self.mode == "files":
                data_length = self.output_size if self.incart else row["data_length"]
            elif self.mode == "memmap":
                data_length = self.memmap_length[row["data"]]
            else:  # npy
                data_length = self.output_size if self.incart else len(self.npy_data[row["data"]])

            if chunk_length == 0:  # do not split
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx, data_length, chunk_length if stride is None else stride))
                idx_end = [min(l + chunk_length, data_length) for l in idx_start]

            # remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if idx_end[i] - idx_start[i] < min_chunk_length:
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            # append to lists
            for _ in range(copies + 1):
                for i_s, i_e in zip(idx_start, idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
        # convert to np.array to avoid mp issues with python lists
        self.df_idx_mapping = np.array(self.df_idx_mapping)
        self.start_idx_mapping = np.array(self.start_idx_mapping)
        self.end_idx_mapping = np.array(self.end_idx_mapping)

    def __len__(self):
        return len(self.df_idx_mapping)

    @property
    def is_empty(self):
        return len(self.df_idx_mapping) == 0

    def __getitem__(self, idx):
        lst = []
        for _ in range(self.sample_items_per_record):
            # determine crop idxs
            timesteps = self.get_sample_length(idx)

            if self.random_crop:  # random crop
                if timesteps == self.output_size:
                    start_idx_rel = 0
                else:
                    start_idx_rel = random.randint(0,
                                                   timesteps - self.output_size - 1)  # np.random.randint(0, timesteps - self.output_size)
            else:
                start_idx_rel = (timesteps - self.output_size) // 2
            if self.sample_items_per_record == 1:
                return self._getitem(idx, start_idx_rel)
            else:
                lst.append(self._getitem(idx, start_idx_rel))
        return tuple(lst)

    def _getitem(self, idx, start_idx_rel):
        # low-level function that actually fetches the data
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        # determine crop idxs
        timesteps = end_idx - start_idx
        assert (timesteps >= self.output_size)
        start_idx_crop = start_idx + start_idx_rel
        end_idx_crop = start_idx_crop + self.output_size

        # print(idx,start_idx,end_idx,start_idx_crop,end_idx_crop)
        # load the actual data
        if (self.mode == "files"):  # from separate files
            if self.incart:
                data_filename = "I" + str(self.timeseries_df_data[df_idx]) + ".npy"
            else:
                data_filename = str(self.timeseries_df_data[df_idx],
                                    encoding='utf-8')  # todo: fix potential issues here
            if self.data_folder is not None:
                if self.incart:
                    data_dir = Path("/mnt/qnap/jeries/incart/processed")
                    data_filename = data_dir / data_filename
                else:
                    data_filename = self.data_folder / data_filename

            # data type has to be adjusted when saving to npy
            data = np.load(data_filename, allow_pickle=True)[start_idx_crop:end_idx_crop]

            ID = data_filename.stem

            if (self.annotation is True):
                label_filename = str(self.timeseries_df_label[df_idx], encoding='utf-8')
                if self.data_folder is not None:
                    label_filename = self.data_folder / label_filename
                label = np.load(label_filename, allow_pickle=True)[
                        start_idx_crop:end_idx_crop]  # data type has to be adjusted when saving to npy
            else:
                label = self.timeseries_df_label[df_idx]  # input type has to be adjusted in the dataframe
        elif (self.mode == "memmap"):  # from one memmap file
            memmap_idx = self.timeseries_df_data[
                df_idx]  # grab the actual index (Note the df to create the ds might be a subset of the original df used to create the memmap)
            memmap_file_idx = self.memmap_file_idx[memmap_idx]
            idx_offset = self.memmap_start[memmap_idx]

            # wi = torch.utils.data.get_worker_info()
            # pid = 0 if wi is None else wi.id#os.getpid()
            # print("idx",idx,"ID",ID,"idx_offset",idx_offset,"start_idx_crop",start_idx_crop,"df_idx", self.df_idx_mapping[idx],"pid",pid)
            mem_filename = str(self.memmap_filenames[memmap_file_idx], encoding='utf-8')
            mem_file = np.memmap(self.memmap_meta_filename.parent / mem_filename, self.memmap_dtype, mode='r',
                                 shape=tuple(self.memmap_shape[memmap_file_idx]))
            data = np.copy(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])  # this is the data sample
            del mem_file
            # print(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            if (self.annotation):
                mem_filename_label = str(self.memmap_filenames_label[memmap_file_idx], encoding='utf-8')
                mem_file_label = np.memmap(self.memmap_meta_filename.parent / mem_filename_label,
                                           self.memmap_dtype_label, mode='r',
                                           shape=tuple(self.memmap_shape_label[memmap_file_idx]))

                label = np.copy(mem_file_label[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
                del mem_file_label
            else:
                label = self.timeseries_df_label[df_idx]
        else:  # single npy array
            ID = self.timeseries_df_data[df_idx]

            data = self.npy_data[ID][start_idx_crop:end_idx_crop]

            if (self.annotation):
                label = self.npy_data_label[ID][start_idx_crop:end_idx_crop]
            else:
                label = self.timeseries_df_label[df_idx]

        sample = (data, label, self.timeseries_df_static[df_idx] if self.static else None)

        if (isinstance(self.transforms, list)):  # transforms passed as list
            for t in self.transforms:
                sample = t(sample)
        elif (self.transforms is not None):  # single transform e.g. from torchvision.transforms.Compose
            sample = self.transforms(sample)

        # consistency check: make sure that data and annotation lengths match
        assert (self.annotation is False or len(sample[0]) == len(sample[1]))

        if (self.static is True):
            return tsdata_static(sample[0], sample[1], sample[2])
        else:
            return tsdata(sample[0] / sample[0].max(), sample[1])

    def get_sampling_weights(self, class_weight_dict, length_weighting=False, timeseries_df_group_by_col=None):
        '''
        class_weight_dict: dictionary of class weights
        length_weighting: weigh samples by length
        timeseries_df_group_by_col: column of the pandas df used to create the object'''
        assert (self.annotation is False)
        assert (length_weighting is False or timeseries_df_group_by_col is None)
        weights = np.zeros(len(self.df_idx_mapping), dtype=np.float32)
        length_per_class = {}
        length_per_group = {}
        for iw, (i, s, e) in enumerate(zip(self.df_idx_mapping, self.start_idx_mapping, self.end_idx_mapping)):
            label = self.timeseries_df_label[i]
            weight = class_weight_dict[label]
            if (length_weighting):
                if label in length_per_class.keys():
                    length_per_class[label] += e - s
                else:
                    length_per_class[label] = e - s
            if (timeseries_df_group_by_col is not None):
                group = timeseries_df_group_by_col[i]
                if group in length_per_group.keys():
                    length_per_group[group] += e - s
                else:
                    length_per_group[group] = e - s
            weights[iw] = weight

        if (length_weighting):  # need second pass to properly take into account the total length per class
            for iw, (i, s, e) in enumerate(zip(self.df_idx_mapping, self.start_idx_mapping, self.end_idx_mapping)):
                label = self.timeseries_df_label[i]
                weights[iw] = (e - s) / length_per_class[label] * weights[iw]
        if (timeseries_df_group_by_col is not None):
            for iw, (i, s, e) in enumerate(zip(self.df_idx_mapping, self.start_idx_mapping, self.end_idx_mapping)):
                group = timeseries_df_group_by_col[i]
                weights[iw] = (e - s) / length_per_group[group] * weights[iw]

        weights = weights / np.min(weights)  # normalize smallest weight to 1
        return weights

    def get_id_mapping(self):
        return self.df_idx_mapping

    def get_sample_id(self, idx):
        return self.df_idx_mapping[idx]

    def get_sample_length(self, idx):
        return self.end_idx_mapping[idx] - self.start_idx_mapping[idx]

    def get_sample_start(self, idx):
        return self.start_idx_mapping[idx]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, transpose_data=True, transpose_label=False):
        # swap channel and time axis for direct application of pytorch's convs
        self.transpose_data = transpose_data
        self.transpose_label = transpose_label

    def __call__(self, sample):

        def _to_tensor(data, transpose=False):
            if (isinstance(data, np.ndarray)):
                if (transpose):  # seq,[x,y,]ch
                    return torch.from_numpy(np.moveaxis(data, -1, 0))
                else:
                    return torch.from_numpy(data)
            else:  # default_collate will take care of it
                return data

        data, label, static = sample

        if not isinstance(data, tuple):
            data = _to_tensor(data, self.transpose_data)
        else:
            data = tuple(_to_tensor(x, self.transpose_data) for x in data)

        if not isinstance(label, tuple):
            label = _to_tensor(label, self.transpose_label)
        else:
            label = tuple(_to_tensor(x, self.transpose_label) for x in label)

        return data, label, static  # returning as a tuple (potentially of lists)


class LongTermDataset(Dataset):

    def __init__(self, data_table, transform, synthetic=False):
        super().__init__()
        self.table = data_table
        self.transform = transform
        self.synthetic = synthetic
        self.x = np.load('/home/jeries.saleh/synth-ecg-analysis/data/x.npy')
        self.y = np.load('/home/jeries.saleh/synth-ecg-analysis/data/y.npy')

    def __getitem__(self, index):
        index = int(index)
        if self.synthetic:
            signal = torch.tensor(self.x[index], dtype=torch.float32)
            label = torch.tensor(self.y[index], dtype=torch.float32)
        else:
            record = wfdb.rdrecord(self.table['file_name'].iloc[index],
                                   sampfrom=self.table['sampfrom'].iloc[index],
                                   sampto=self.table['sampto'].iloc[index],
                                   channels=[0]  # Use only the first channel (ECG lead) out of the two available
                                   )

            signal = self.transform(record.p_signal)
            label = self.table['label'].iloc[index]
        return signal, label

    def __len__(self):
        return len(self.table) if not self.synthetic else len(self.y)

# class LongTermDataset(Dataset):
#     def __init__(self, data_table, transform=None, scaling_method='mean', num_bins=10):
#         super().__init__()
#         self.table = data_table
#         self.transform = transform
#         self.scaling_method = scaling_method
#         self.num_bins = num_bins
#
#     def __getitem__(self, index):
#         index = int(index)  # In case given as Torch type, convert to a regular integer type.
#
#         record = wfdb.rdrecord(self.table['file_name'].iloc[index],
#                                sampfrom=self.table['sampfrom'].iloc[index],
#                                sampto=self.table['sampto'].iloc[index],
#                                channels=[0])  # Use only the first channel (ECG lead)
#
#         signal = record.p_signal.flatten()  # Flatten in case it is 2D (single channel)
#
#         # Apply any additional transforms specified
#         if self.transform:
#             signal = self.transform(signal)
#
#         # Scale the signal
#         signal = self.scale_time_series(signal.detach().numpy())
#
#         # Quantize the signal
#         signal = self.quantize_time_series(signal)
#
#         label = int(self.table['label'].iloc[index])
#
#         return signal, label
#
#     def __len__(self):
#         return len(self.table)
#
#     def scale_time_series(self, signal):
#         """
#         Scales a 1D time series signal.
#         """
#         if self.scaling_method == 'mean':
#             mean = np.mean(signal)
#             mad = np.mean(np.abs(signal - mean))
#             scaled_signal = (signal - mean) / mad
#
#         elif self.scaling_method == 'minmax':
#             min_val = np.min(signal)
#             max_val = np.max(signal)
#             scaled_signal = (signal - min_val) / (max_val - min_val)
#
#         elif self.scaling_method == 'standard':
#             mean = np.mean(signal)
#             std = np.std(signal)
#             scaled_signal = (signal - mean) / std
#
#         else:
#             raise ValueError("Unsupported scaling method. Choose 'mean', 'minmax', or 'standard'.")
#
#         return scaled_signal
#
#     def quantize_time_series(self, signal):
#         """
#         Quantizes a 1D time series signal into discrete bins.
#         """
#         bins = np.linspace(np.min(signal), np.max(signal), self.num_bins + 1)
#         quantized_signal = np.digitize(signal, bins) - 1  # `-1` to make bins 0-indexed
#         return quantized_signal


class RRLongTerm(Dataset):

    def __init__(self, data_table):
        super().__init__()
        self.table = data_table

    def __getitem__(self, index):
        index = int(index)

        annotation_samples = wfdb.rdann(self.table['file_name'].iloc[index],
                                        'atr',
                                        sampfrom=self.table['sampfrom'].iloc[index],
                                        sampto=self.table['sampto'].iloc[index],
                                        shift_samps=True
                                        )

        rr_intervals = np.diff(annotation_samples.sample.astype(np.int32))

        label = self.table['label'].iloc[index]
        return rr_intervals, label

    def __len__(self):
        return len(self.table)