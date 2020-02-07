import argparse
import logging
import os
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from pprint import pprint


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--device', type=int, default=None)
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

    datasets, ontology, vocab, E = load_dataset()
    idxs, seed_idxs, num_turns = datasets["train"].get_turn_idxs(
        100, 0, sample_mode="singlepass")

    if args.model == "glad":
        model = GLAD(args, ontology, vocab)
    elif args.model == "gce":
        model = GCE(args, ontology, vocab)
    else:
        raise NotImplementedError()

    if model.optimizer is None:
        model.set_optimizer()

    iteration = 0
    for epoch in range(args.epochs):
        logging.info('starting epoch {}'.format(epoch))

        # train and update parameters
        for batch, batch_labels in datasets["train"].batch(
                batch_size=args.batch_size,
                idxs=idxs,
                labels=datasets["train"].get_labels(),
                shuffle=True):
            iteration += 1
            model.zero_grad()
            loss, scores = model.forward(batch, batch_labels, training=True)
            loss.backward()
            model.optimizer.step()

    logging.info('Running dev evaluation')
    dev_out = model.run_eval(datasets["test"], args)
    pprint(dev_out)


if __name__ == "__main__":
    main()
