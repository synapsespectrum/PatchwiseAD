from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
import torch.multiprocessing
from models import AnomalyBERT, TPADFormer
from utils.losses import BCELoss, PretrainingLoss
from utils.compute_metrics import f1_score
from timm.scheduler.cosine_lr import CosineLRScheduler

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')


class Experiment(Exp_Basic):
    def __init__(self, args):
        super(Experiment, self).__init__(args)

    def _build_model(self, n_vars):
        self.args.n_vars = n_vars
        self.args.dim_output = 1 if self.args.loss == 'BCE' else n_vars
        model_dict = {
            'AnomalyBERT': AnomalyBERT,
            'TFADFormer': TPADFormer
        }

        model = model_dict[self.args.model].Model(self.args)

        # Summary of the model
        summary(model, input_size=(self.args.batch_size, self.args.input_encoder_len * self.args.patch_size, n_vars))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.is_pretrain:
            criterion = PretrainingLoss()
        elif self.args.loss == 'BCE':
            criterion = BCELoss()  # Custom BCE loss function
        else:
            criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        self.model = self._build_model(n_vars=train_data.n_vars).to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = CosineLRScheduler(model_optim,
                                      t_initial=self.args.train_epochs,
                                      lr_min=self.args.learning_rate * 0.01,
                                      warmup_lr_init=self.args.learning_rate * 0.001,
                                      warmup_t=self.args.train_epochs // 10,
                                      cycle_limit=1,
                                      t_in_epochs=False,
                                      )

        # # Load a checkpoint if exists.
        # if self.args.checkpoints != None:
        #     self.model.load_state_dict(torch.load(self.args.checkpoints, map_location='cpu'))
        #
        if self.args.logs is None:
            if not os.path.exists('./logs/'):
                os.mkdir('./logs/')
            log_dir = os.path.join('./logs/',
                                   time.strftime('%y%m%d%H%M%S_' + self.args.dataset, time.localtime(time.time())))
            os.mkdir(log_dir)
            os.mkdir(os.path.join(log_dir, 'state'))
        else:
            log_dir = self.args.logs
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
                os.mkdir(os.path.join(log_dir, 'state'))

        # hyperparameters save
        with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

        torch.save(self.model, os.path.join(log_dir, 'model.pt'))

        self.summary_writer = SummaryWriter(log_dir)
        print("Starting training with iteration: ", train_steps)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            loop_span = time.time()
            for i, (batch_x, labels) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                if self.args.is_pretrain:
                    outputs, labels, mask_map = self.model(batch_x)
                    # calculate loss
                    loss = criterion(outputs, labels, mask_map)
                else:
                    outputs = self.model(batch_x)
                    outputs = outputs.squeeze(-1)
                    # calculate loss
                    loss = criterion(outputs, labels)

                if (i % self.args.summary_steps) == 0 and not self.args.is_pretrain:  # Checking the training process
                    with torch.no_grad():
                        pred = (nn.Sigmoid()(outputs) > 0.5).int()
                        total_data_num = labels.size(0) * labels.size(1)  # batch_size * seq_len
                        acc = (pred == labels.int()).sum().item() / total_data_num
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | train accuracy: {3:.10f}".
                              format(i, epoch, loss.item(), acc))
                        num_iters = epoch * train_steps + i
                        self.summary_writer.add_scalar('Train/Loss', loss.item(), num_iters)
                        self.summary_writer.add_scalar('Train/Accuracy', acc, num_iters)

                        # validation
                        print("Validating the model...")
                        t1 = time.time()
                        best_eval, best_rate = self.validate(test_data)
                        print(f'anomaly rate: {best_rate:.3f} | '
                              f'precision: {best_eval[0]:.5f} | '
                              f'recall: {best_eval[1]:.5f} | '
                              f'F1-score: {best_eval[2]:.5f}')
                        print(f"Finishing Validating within: ", time.time() - t1)

                        self.summary_writer.add_scalar('Valid/Best Anomaly Rate', best_rate, num_iters)
                        self.summary_writer.add_scalar('Valid/Precision', best_eval[0], num_iters)
                        self.summary_writer.add_scalar('Valid/Recall', best_eval[1], num_iters)
                        self.summary_writer.add_scalar('Valid/F1', best_eval[2], num_iters)

                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                        print(f"Total time for {self.args.summary_steps} iteration: ", time.time() - loop_span, "\n")
                        loop_span = time.time()
                    torch.save(self.model.state_dict(),
                               os.path.join(log_dir, f"state/state_dict_step_epoch_{epoch}_iter{i}.pt"))

                if self.args.is_pretrain and (i % self.args.summary_steps) == 0:  # Logging the training process
                    num_iters = epoch * train_steps + i
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i, epoch, loss.item()))
                    print(f"Time for {self.args.summary_steps} iteration: ", time.time() - loop_span, "\n")
                    loop_span = time.time()
                    self.summary_writer.add_scalar('Train/Loss', loss.item(), num_iters)
                    # if the loss is best, save the model
                    if loss.item() < min(train_loss):
                        torch.save(self.model.state_dict(),
                                   os.path.join(log_dir, f"state/state_dict_step_epoch_{epoch}_iter{i}.pt"))

                train_loss.append(loss.item())

                # Update the model parameters
                model_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
                model_optim.step()

                scheduler.step_update(num_iters)

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            print("Epoch: ", epoch, " | Loss: ", np.mean(train_loss), " | Time: ", time.time() - epoch_time)

        torch.save(self.model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))
        torch.save(self.model, os.path.join(log_dir, 'model.pt'))
        print("Saving the model to ", os.path.join(log_dir, 'state_dict.pt'))

        return self.model

    def validate(self, test_data):
        data = test_data.data
        if self.model is None:
            raise ValueError('Model is not loaded.')
        self.model.eval()
        test_estimation = self.estimate(test_data, data)
        test_estimation = test_estimation[:, 0].cpu().numpy()

        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(0.001, 0.301, 0.001):
            evaluation = f1_score(test_data.labels, test_estimation, rate, False, False)
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate

        return best_eval, best_rate

    @torch.no_grad()
    def estimate(self, test_data, data, check_count=None):
        window_size = self.model.input_encoder_len * self.model.patch_size
        assert window_size % self.args.window_sliding == 0

        n_column = 1 if self.args.loss == 'BCE' else data.shape[1]
        n_batch = self.args.batch_size
        batch_sliding = n_batch * window_size
        _batch_sliding = n_batch * self.args.window_sliding

        # Pre-allocate output tensor on GPU
        output_values = torch.zeros(len(data), n_column, device=self.device)
        count = 0
        checked_index = float('inf') if check_count is None else check_count

        # Move data to GPU once
        data = data.to(self.device)

        sigmoid = nn.Sigmoid()

        for division in test_data.divisions:
            data_len = division[1] - division[0]
            last_window = data_len - window_size + 1
            _test_data = data[division[0]:division[1]]

            # Preallocate division-specific output tensors
            _output_values = torch.zeros(data_len, n_column, device=self.device)
            n_overlap = torch.zeros(data_len, device=self.device)

            _first = -batch_sliding
            # First loop: Process large batches
            for first in range(0, last_window - batch_sliding + 1, batch_sliding):
                for i in range(first, first + window_size, self.args.window_sliding):
                    x = _test_data[i:i + batch_sliding].reshape(n_batch, window_size, -1)

                    # Model forward pass
                    y = sigmoid(self.model(x))

                    # In-place accumulation of output and overlap counts
                    _output_values[i:i + batch_sliding].add_(y.view(-1, n_column))
                    n_overlap[i:i + batch_sliding].add_(1)

                    count += n_batch

                    if count > checked_index:
                        print(count, 'windows are computed.')
                        checked_index += check_count

                _first = first

            _first += batch_sliding

            # Second loop: Process overlapping windows
            for first, last in zip(range(_first, last_window, _batch_sliding),
                                   list(range(_first + _batch_sliding, last_window, _batch_sliding)) + [last_window]):
                x_list = []
                for i in list(range(first, last - 1, self.args.window_sliding)) + [last - 1]:
                    x_list.append(_test_data[i:i + window_size])

                x = torch.stack(x_list)

                # Model forward pass
                y = sigmoid(self.model(x))

                # In-place accumulation for overlapping windows
                for i, j in enumerate(list(range(first, last - 1, self.args.window_sliding)) + [last - 1]):
                    _output_values[j:j + window_size].add_(y[i])
                    n_overlap[j:j + window_size].add_(1)

                count += n_batch

                if count > checked_index:
                    print(count, 'windows are computed.')
                    checked_index += check_count

            # Use in-place division
            _output_values.div_(n_overlap.unsqueeze(-1))
            output_values[division[0]:division[1]] = _output_values

        return output_values
