import argparse
import random

import numpy as np
import torch

from utils.print_args import print_args
from utils.arg_parser import create_parser

from exp.exp_main import Experiment


def set_seed(seed: int = 2024):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_experiment(args: argparse.Namespace, setting: str):
    """Run a single experiment with the given args and setting."""
    exp = Experiment(args)

    if args.is_training:
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

    torch.cuda.empty_cache()


def generate_setting(args: argparse.Namespace, iteration: int) -> str:
    """Generate a setting string based on the args and iteration."""
    return f'{args.model}_{args.dataset}_{args.input_encoder_len}_ps{args.patch_size}_n-heads{args.n_heads}' \
           f'_e-layers{args.e_layers}_dmodels{args.d_model}_dr{args.dropout}_{args.replacing_rate_max}_' \
           f'{args.replacing_weight}_{iteration} '


def main():
    set_seed()
    parser = create_parser()
    args = parser.parse_args()

    print("CUDA:", torch.cuda.is_available())
    print('Args in experiment:')
    print_args(args)

    if args.is_training:
        for i in range(args.itr):
            setting = generate_setting(args, i)
            run_experiment(args, setting)
    else:
        setting = generate_setting(args, 0)
        run_experiment(args, setting)


if __name__ == '__main__':
    main()
