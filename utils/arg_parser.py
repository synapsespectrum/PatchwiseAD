import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description='Anomaly Detection')

    # Basic config
    parser.add_argument('--is_pretrain', type=int, required=True, default=0, help='pretrain status')
    parser.add_argument('--is_finetune', type=int, required=True, default=1, help='finetune status')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--pretrain_model_path', type=str, default=None, help='pretrained model path')
    parser.add_argument('--model', type=str, required=True, default='AnomalyBERT',
                        help='model name, options: [AnomalyBERT, TDAnomalyBERT]')

    # Data loader
    parser.add_argument("--dataset", required=True, type=str, help='SMAP/MSL/SMD/SWaT/WADI')
    parser.add_argument("--data_path", default='./datasets/', type=str, help='path to data')
    parser.add_argument("--window_sliding", default=16, type=int, help='sliding steps of windows for validation')
    parser.add_argument("--data_division", default=None, type=str,
                        help='data division for validation; None(default)/channel/class/total')
    parser.add_argument("--replacing_data", default=None, type=str,
                        help='external data for soft replacement; None(default)/SMAP/MSL/SMD/SWaT/WADI')

    # Data augmentation
    parser.add_argument("--augment", default=1, type=int, help='data augmentation')
    parser.add_argument("--soft_replacing", default=0.5, type=float, help='probability for soft replacement')
    parser.add_argument("--flip_replacing_interval", default='all', type=str,
                        help='allowance for random flipping in soft replacement; vertical/horizontal/all/none')
    parser.add_argument("--uniform_replacing", default=0.15, type=float, help='probability for uniform replacement')
    parser.add_argument("--peak_noising", default=0.15, type=float, help='probability for peak noise')
    parser.add_argument("--length_adjusting", default=0.0, type=float, help='probability for length adjustment')
    parser.add_argument("--white_noising", default=0.0, type=float, help='probability for white noise (deprecated)')

    # Experimental log
    parser.add_argument("--summary_steps", default=500, type=int,
                        help='steps for summarizing and saving of training log')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--logs', type=str, default=None, help='location of logs')

    # Anomaly detection config
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # Reconstruction config
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='mask ratio (%)')

    # Model define
    parser.add_argument("--input_encoder_len", default=512, type=int, help='number of features for a window')
    parser.add_argument("--patch_size", default=4, type=int, help='number of data points in a patch')
    parser.add_argument('--overlap', type=int, default=0, help='overlap size for patch embedding')
    # Input sequence len = input_encoder_length * patch_size
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--positional_encoding', type=str, default='None',
                        help='positional encoding for embedded input; None/Sinusoidal/Absolute')
    parser.add_argument('--relative_position_embedding', type=int, default=1,
                        help='relative position embedding option')
    parser.add_argument('--hidden_dim_rate', type=float, default=4., help='hidden layer dimension rate to d_model')

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--replacing_rate_max", default=0.15, type=float,
                        help='maximum ratio of replacing interval length to window size')
    parser.add_argument("--replacing_weight", default=0.7, type=float,
                        help='weight for external interval in soft replacement')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    # Optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times for each setting')
    parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--loss', type=str, default='BCE', help='loss function')
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)

    # GPU
    parser.add_argument("--use_gpu", default=1, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # De-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    return parser
