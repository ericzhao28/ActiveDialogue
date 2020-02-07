import argparse
import os
from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--al_batch', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=10000)
    parser.add_argument('--seed_size', type=int, default=1000)
    parser.add_argument('--sample_mode', type=str, default="singlepass")
    parser.add_argument('--recency_bias', type=int, default=3)

    parser.add_argument('--model', type=str, default='glad')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=50)
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
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-3,
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

    args = parser.parse_args()
    args.dout = os.path.join(args.dexp, args.model, args.nick)
    args.dropout = {
        d.split('=')[0]: float(d.split('=')[1]) for d in args.dropout
    }
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    env = DSTEnv(load_dataset, GLAD, args)
    if args.seed:
        env.train_seed()
    ended = False
    while not ended:
        raw_obs, obs_dist = env.observe()
        ended = env.step()

    env.eval()


if __name__ == "__main__":
    main()

