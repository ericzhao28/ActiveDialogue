import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--al_batch', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=30000)
    parser.add_argument('--label_budget', type=int, default=100000)
    parser.add_argument('--seed_size', type=int, default=200)
    parser.add_argument('--sample_mode', type=str, default="singlepass")
    parser.add_argument('--recency_bias', type=int, default=1)
    parser.add_argument('--fit_items', type=int, default=512)
    parser.add_argument('--eval_period', type=int, default=8)
    parser.add_argument('--model', type=str, default='glad')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed_epochs', type=int, default=40)
    parser.add_argument('--strategy', type=str, default="")
    parser.add_argument('--label_timeout', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--comp_batch_size', type=int, default=32)
    parser.add_argument('--dexp',
                        help='root experiment folder',
                        default='exp')
    parser.add_argument('--demb',
                        help='word embedding size',
                        default=400,
                        type=int)
    parser.add_argument('--dhid',
                        help='hidden state size',
                        default=200,
                        type=int)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--lr',
                        help='learning rate',
                        default=2e-3,
                        type=float)
    parser.add_argument('--stop',
                        help='slot to early stop on',
                        default='joint_goal')
    parser.add_argument('--resume', help='save directory to resume from')
    parser.add_argument('-n',
                        '--nick',
                        help='nickname for model',
                        default='default')
    parser.add_argument('--dropout',
                        nargs='*',
                        help='dropout rates',
                        default=['emb=0.2', 'local=0.2', 'global=0.2'])
    parser.add_argument('--sl_reduction',
                        dest='sl_reduction',
                        action='store_true')
    parser.set_defaults(sl_reduction=False)
    parser.add_argument('--optimistic_weighting',
                        dest='optimistic_weighting',
                        action='store_true')
    parser.set_defaults(optimistic_weighting=False)
    parser.set_defaults(device=None)

    args = parser.parse_args()
    args.dout = os.path.join(args.dexp, args.model, args.nick)
    args.dropout = {
        d.split('=')[0]: float(d.split('=')[1]) for d in args.dropout
    }
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    return args
