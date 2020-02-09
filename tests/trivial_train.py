import logging
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from pprint import pprint


def main():
    args = get_args()

    datasets, ontology, vocab, E = load_dataset()
    ptrs, seed_ptrs, num_turns = datasets["train"].get_turn_ptrs(
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
                ptrs=ptrs,
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
