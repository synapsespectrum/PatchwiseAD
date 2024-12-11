import numpy as np
import torch
from torch.utils.data import Dataset
from data_provider import config


class AnomalyDataAugmenter:
    def __init__(self, args, numerical_column, categorical_column, replacing_data, full_data):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_seq_len = args.input_encoder_len * args.patch_size
        self.replacing_table_length = 10000
        self.replacing_data = replacing_data
        self.full_data = full_data

        # Anomaly generation parameters
        replacing_rate = (args.replacing_rate_max / 10, args.replacing_rate_max)
        self.replacing_len_max = int(args.replacing_rate_max * self.data_seq_len)
        # Initialize replacing table
        self.replacing_table = list(
            np.random.randint(int(self.data_seq_len * replacing_rate[0]),
                              int(self.data_seq_len * replacing_rate[1]),
                              size=self.replacing_table_length))

        self.replacing_table_index = 0

        self.numerical_column = numerical_column
        self.num_numerical = len(self.numerical_column)
        self.categorical_column = categorical_column
        self.num_categorical = len(self.categorical_column)

        # Synthesis probability
        self.soft_replacing_prob = 1 - args.soft_replacing
        self.uniform_replacing_prob = self.soft_replacing_prob - args.uniform_replacing
        self.peak_noising_prob = self.uniform_replacing_prob - args.peak_noising
        self.length_adjusting_prob = self.peak_noising_prob - args.length_adjusting \
            if args.loss == 'BCE' else self.peak_noising_prob
        self.white_noising_prob = args.white_noising

        # Soft replacing flip options
        if args.flip_replacing_interval == 'all':
            self.vertical_flip = True
            self.horizontal_flip = True
        elif args.flip_replacing_interval == 'vertical':
            self.vertical_flip = True
            self.horizontal_flip = False
        elif args.flip_replacing_interval == 'horizontal':
            self.vertical_flip = False
            self.horizontal_flip = True
        elif args.flip_replacing_interval == 'none':
            self.vertical_flip = False
            self.horizontal_flip = False

    def __replacing_weights(self, interval_len):
        warmup_len = interval_len // 10
        return np.concatenate((np.linspace(0, self.args.replacing_weight, num=warmup_len),
                               np.full(interval_len - 2 * warmup_len, self.args.replacing_weight),
                               np.linspace(self.args.replacing_weight, 0, num=warmup_len)), axis=None)

    def augment_batch(self, batch_x):
        x, indexes = batch_x
        n_batch, data_seq_len, n_vars = x.shape
        # Get replacing lengths
        replacing_lengths = self._get_replacing_lengths(n_batch)

        # Generate replacing indices and targets
        replacing_index = np.random.randint(0, (len(self.replacing_data) - replacing_lengths + 1)[:, np.newaxis],
                                            size=(n_batch, n_vars))
        target_index = np.random.randint(0, data_seq_len - replacing_lengths + 1)

        # Generate replacing types and dimensions
        replacing_type = np.random.uniform(0., 1., size=(n_batch,))
        replacing_dim_numerical = self._generate_replacing_dim(n_batch, self.num_numerical)
        replacing_dim_categorical = self._generate_replacing_dim(n_batch, self.num_categorical)

        x_anomaly = torch.zeros(n_batch, data_seq_len, device=self.device)

        for j in range(n_batch):
            x[j], x_anomaly[j] = self._augment_single_sample(
                x[j], replacing_index[j], target_index[j], replacing_lengths[j],
                replacing_type[j], replacing_dim_numerical[j], replacing_dim_categorical[j],
                indexes[j]
            )

        return x, x_anomaly  # augmented_data, anomaly_labels

    def _get_replacing_lengths(self, n_batch):
        current_index = self.replacing_table_index
        self.replacing_table_index += n_batch

        if self.replacing_table_index > len(self.replacing_table):
            replacing_lengths = self.replacing_table[current_index:]
            self.replacing_table_index -= len(self.replacing_table)
            replacing_lengths += self.replacing_table[:self.replacing_table_index]
        else:
            replacing_lengths = self.replacing_table[current_index:self.replacing_table_index]
            if self.replacing_table_index == len(self.replacing_table):
                self.replacing_table_index = 0

        return np.array(replacing_lengths)

    def _generate_replacing_dim(self, n_batch, num_features):
        dim = np.random.uniform(0., 1., size=(n_batch, num_features))
        return dim - np.maximum(dim.min(axis=1, keepdims=True), 0.3) <= 0.001

    def _augment_single_sample(self, x, rep, tar, leng, typ, dim_num, dim_cat, index):
        if leng == 0:
            return x, torch.zeros_like(x[:, 0])

        x_anomaly = torch.zeros_like(x[:, 0])  # shape = (data_seq_len)
        x_rep = x[tar:tar + leng].clone()
        _x = x_rep.clone().transpose(0, 1).to(self.device)

        rep_len_num = len(dim_num[dim_num])
        rep_len_cat = len(dim_cat[dim_cat]) if len(dim_cat) > 0 else 0
        target_column_numerical = self.numerical_column[dim_num]
        target_column_categorical = self.categorical_column[dim_cat] if rep_len_cat > 0 else []

        # External interval replacing
        if typ > self.soft_replacing_prob:
            _x = self._external_interval_replacing(_x, self.replacing_data, rep, leng,
                                                   rep_len_num,
                                                   rep_len_cat,
                                                   target_column_numerical,
                                                   target_column_categorical)
        elif typ > self.uniform_replacing_prob:
            _x = self._uniform_replacing(_x, target_column_numerical, rep_len_num)
        elif typ > self.peak_noising_prob:
            _x, x_anomaly = self._peak_noising(_x, x_anomaly, target_column_numerical, leng, tar, rep_len_num)
            x[tar:tar + leng] = _x.transpose(0, 1)
            return x, x_anomaly
        elif typ > self.length_adjusting_prob:
            x, x_anomaly = self._length_adjusting(x, tar, leng, index=index)
            return x, x_anomaly
        elif typ < self.white_noising_prob:
            _x = self._white_noising(_x, leng, target_column_numerical)
        else:
            return x, x_anomaly

        x_anomaly[tar:tar + leng] = 1
        x[tar:tar + leng] = _x.transpose(0, 1)
        return x, x_anomaly

    def _external_interval_replacing(self, _x, replacing_data, rep, leng,
                                     rep_len_num,
                                     rep_len_cat,
                                     target_column_numerical,
                                     target_column_categorical):
        # Replacing for numerical columns
        _x_temp = []
        col_num = np.random.choice(self.numerical_column, size=rep_len_num)
        filp = np.random.randint(0, 2, size=(rep_len_num, 2)) > 0.5
        for _col, _rep, _flip in zip(col_num, rep[:rep_len_num], filp):
            random_interval = replacing_data[_rep:_rep + leng, _col].copy()
            # fliping options
            if self.horizontal_flip and _flip[0]:
                random_interval = random_interval[::-1].copy()
            if self.vertical_flip and _flip[1]:
                random_interval = 1 - random_interval
            _x_temp.append(torch.from_numpy(random_interval))
        _x_temp = torch.stack(_x_temp).to(self.device)
        weights = torch.from_numpy(self.__replacing_weights(leng)).float().unsqueeze(0).to(self.device)
        _x[target_column_numerical] = _x_temp * weights + _x[target_column_numerical] * (1 - weights)

        # Replacing for categorical columns
        if rep_len_cat > 0:
            _x_temp = []
            col_cat = np.random.choice(self.categorical_column, size=rep_len_cat)
            for _col, _rep in zip(col_cat, rep[-rep_len_cat:]):
                _x_temp.append(torch.from_numpy(replacing_data[_rep:_rep + leng, _col].copy()))
            _x_temp = torch.stack(_x_temp).to(self.device)
            _x[target_column_categorical] = _x_temp

        return _x

    def _uniform_replacing(self, _x, target_column_numerical, rep_len_num):
        _x[target_column_numerical] = torch.rand(rep_len_num, 1, device=self.device)
        return _x

    def _peak_noising(self, _x, x_anomaly, target_column_numerical, leng, tar, rep_len_num):
        peak_index = np.random.randint(0, leng)
        peak_value = (_x[target_column_numerical, peak_index] < 0.5).float().to(self.device)
        peak_value = peak_value + (0.1 * (1 - 2 * peak_value)) * torch.rand(rep_len_num, device=self.device)
        _x[target_column_numerical, peak_index] = peak_value

        peak_index = tar + peak_index
        # Calculate the range of affected indices
        tar_first = np.maximum(0, peak_index - self.args.patch_size)
        tar_last = peak_index + self.args.patch_size + 1
        x_anomaly[tar_first:tar_last] = 1
        return _x, x_anomaly

    def _length_adjusting(self, x, tar, leng, index):
        x_anomaly = torch.zeros_like(x[:, 0])
        if leng > self.replacing_len_max // 2:
            # Lengthening
            scale = np.random.randint(2, 5)
            _leng = leng - leng % scale
            scaled_leng = _leng // scale
            x[tar + _leng:] = x[tar + scaled_leng:-_leng + scaled_leng].clone()
            x[tar:tar + _leng] = torch.repeat_interleave(x[tar:tar + scaled_leng], scale, dim=0)
            x_anomaly[tar:tar + _leng] = 1
        else:
            # Shortening
            if index > self.replacing_len_max * 1.5:
                scale = np.random.randint(2, 5)
                _leng = leng * (scale - 1)
                x[:tar] = torch.from_numpy(self.full_data[index - _leng:index + tar - _leng]).float().to(
                    self.device)
                x[tar:tar + leng] = torch.from_numpy(
                    self.full_data[index + tar - _leng:index + tar + leng:scale]).float().to(self.device)
                x_anomaly[tar:tar + leng] = 1
        return x, x_anomaly

    def _white_noising(self, _x, leng, target_column_numerical):
        # Apply white noise to numerical columns
        noise = torch.normal(mean=0., std=0.003, size=(len(target_column_numerical), leng), device=self.device)
        _x[target_column_numerical] = (_x[target_column_numerical] + noise).clamp(min=0., max=1.)

        return _x


class AnomalyDataset(Dataset):
    def __init__(self, args, flag='train'):
        self.dataset_name = args.dataset
        self.flag = flag
        self.input_encoder_length = args.input_encoder_len
        self.patch_size = args.patch_size
        self.data_seq_len = self.input_encoder_length * self.patch_size

        if flag == 'train':
            self.data = np.load(config.TRAIN_DATASET[self.dataset_name]).copy().astype(np.float32)
            self.replacing_data = self.data if args.replacing_data is None else np.load(
                config.TRAIN_DATASET[args.replacing_data]).copy().astype(np.float32)
        elif flag == 'test':
            self.data = torch.Tensor(np.load(config.TEST_DATASET[self.dataset_name]).copy().astype(np.float32))
            self.labels = np.load(config.TEST_LABEL[self.dataset_name]).copy().astype(np.int32)
            # Data division for test
            self.data_division = config.DEFAULT_DIVISION[args.dataset] if args.data_division == None \
                else args.data_division
            if self.data_division == 'total':
                self.divisions = [[0, len(self.data)]]
            else:
                import json
                with open(config.DATA_DIVISION[args.dataset][self.data_division], 'r') as f:
                    self.divisions = json.load(f)
                if isinstance(self.divisions, dict):
                    self.divisions = self.divisions.values()

        else:
            raise ValueError("flag must be one of 'train', 'val', or 'test'")

        self.n_vars = self.data.shape[1] # number of variables

        self.numerical_column = np.array(config.NUMERICAL_COLUMNS[self.dataset_name])
        self.categorical_column = np.array(config.CATEGORICAL_COLUMNS[self.dataset_name])
        self._preprocess_data()

    def _preprocess_data(self):
        if self.dataset_name in config.IGNORED_COLUMNS:
            ignored_column = np.array(config.IGNORED_COLUMNS[self.dataset_name])
            remaining_column = [col for col in range(self.n_vars) if col not in ignored_column]
            self.data = self.data[:, remaining_column]
            if self.flag == 'train':
                self.replacing_data = self.replacing_data[:, remaining_column]

            self.n_vars = len(remaining_column)
            self.numerical_column -= (self.numerical_column[:, None] - ignored_column[None, :] > 0).astype(int).sum(
                axis=1)
            self.categorical_column -= (self.categorical_column[:, None] - ignored_column[None, :] > 0).astype(int).sum(
                axis=1)

    def __len__(self):
        return len(self.data) - self.data_seq_len + 1

    def __getitem__(self, index):
        if self.flag == 'train':
            x = torch.tensor(self.data[index:index + self.data_seq_len])
            return x, index
        elif self.flag == 'test':
            x = torch.tensor(self.data[index:index + self.data_seq_len])
            y = torch.tensor(self.labels[index:index + self.data_seq_len])
            return x, y

    def get_replacing_data(self, length):
        start_index = np.random.randint(0, len(self.replacing_data) - length + 1)
        return self.replacing_data[start_index:start_index + length]
