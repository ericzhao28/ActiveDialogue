import logging
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE


def main():
    args = get_args()
    args.epoch = 2
    args.batch_size = 32
    datasets, ontology, vocab, E = load_dataset()
    ptrs, seed_ptrs, num_turns = datasets["dev"].get_turn_ptrs(
        1, 10, sample_mode="singlepass")
    for model in [GLAD(args, ontology, vocab), GCE(args, ontology, vocab)]:
        if model.optimizer is None:
            model.set_optimizer()
        iteration = 0
        for epoch in range(args.epochs):
            logging.info('starting epoch {}'.format(epoch))
            for batch, batch_labels in datasets["dev"].batch(
                    batch_size=args.batch_size,
                    ptrs=ptrs,
                    shuffle=True):
                iteration += 1
                model.zero_grad()
                loss, scores = model.forward(batch, batch_labels, training=True)
                loss.backward()
                model.optimizer.step()
                if iteration > 2:
                    iteration = 0
                    break
        dev_out = model.run_eval(datasets["dev"], args)


if __name__ == "__main__":
    main()
