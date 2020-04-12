import argparse
import os
import shlex


def get_args(cmd):
    parser = argparse.ArgumentParser()

    # Common Settings
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--strategy', type=str, default="")
    parser.add_argument('--model', type=str, default='glad')
    parser.add_argument('--init_threshold', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.75)

    # General hyperparameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--final_epochs', type=int, default=50)
    parser.add_argument('--seed_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed_batch_size', type=int, default=128)
    parser.add_argument('--comp_batch_size', type=int, default=128)
    parser.add_argument('--inference_batch_size', type=int, default=512)

    # Practical settings
    parser.add_argument('--noise_fn', type=float, default=0.0)
    parser.add_argument('--noise_fp', type=float, default=0.0)

    # AL Setting
    parser.add_argument('--al_batch', type=int, default=64)
    parser.add_argument('--label_budget', type=int, default=160)
    parser.add_argument('--seed_size', type=int, default=1000)
    parser.add_argument('--sample_mode', type=str, default="singlepass")
    parser.add_argument('--num_passes', type=int, default=1)

    # Stable
    parser.add_argument('--eval_period', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.set_defaults(device=None)
    parser.add_argument('--force_seed',
                        dest='force_seed',
                        action='store_true')
    parser.set_defaults(force_seed=False)

    # Frozen threshold
    parser.add_argument('--threshold_strategy', type=str, default="fixed")
    parser.add_argument('--threshold_scaler', type=float, default=0.95)
    parser.add_argument(
        '--rejection_ratio',
        type=float,
        default=16.0)  # for thresholds
    parser.add_argument('--threshold_noise_std', type=float, default=0.05)

    # Frozen hyperparams
    parser.add_argument('--demb',
                        help='word embedding size',
                        default=400,
                        type=int)
    parser.add_argument('--dhid',
                        help='hidden state size',
                        default=200,
                        type=int)
    parser.add_argument('--dropout',
                        nargs='*',
                        help='dropout rates',
                        default=['emb=0.2', 'local=0.2', 'global=0.2'])
    parser.add_argument('--stop',
                        help='slot to early stop on',
                        default='joint_goal')

    # Frozen logistics
    parser.add_argument('--resume', help='save directory to resume from')
    parser.add_argument('-n',
                        '--nick',
                        help='nickname for model',
                        default='default')
    parser.add_argument('--dexp',
                        help='root experiment folder',
                        default='exp')

    # Outdated (bags)
    parser.add_argument('--fit_items', type=int, default=512)
    parser.add_argument('--label_timeout', type=int, default=10)
    parser.add_argument(
        "-f",
        "--fff",
        help="a dummy argument to fool ipython",
        default="1")
    parser.add_argument('--sl_reduction',
                        dest='sl_reduction',
                        action='store_true')
    parser.set_defaults(sl_reduction=False)
    parser.add_argument('--optimistic_weighting',
                        dest='optimistic_weighting',
                        action='store_true')
    parser.set_defaults(optimistic_weighting=False)

    # Parse arguments
    if cmd:
        args = parser.parse_args(shlex.split(cmd))
    else:
        args = parser.parse_args()

    args.dout = os.path.join(args.dexp, args.nick)
    args.dropout = {
        d.split('=')[0]: float(d.split('=')[1]) for d in args.dropout
    }
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    return args
