"""
Run seed training experiments (passive learning).
"""

import sys
import logging
from comet_ml import Experiment
import torch
from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key


def main(cmd=None, stdout=True):
    """Run seed training for fixed seed."""

    # Initialize system
    args = get_args(cmd)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.device)

    # Initialize logging
    model_id = "Base {}, seed size {}, epochs {}, batch size {}, lr {}".format(
        args.model, args.seed_size, args.epochs, args.batch_size, args.lr)
    logging.basicConfig(
        filename="{}/{}.txt".format(args.dout, model_id),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    if stdout:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = Experiment(comet_ml_key, project_name="ActiveDialogue")
    logger.set_name(model_id)
    logger.log_parameters(vars(args))

    # Select model and environment
    if args.model == "glad":
        model_arch = GLAD
    elif args.model == "gce":
        model_arch = GCE
    env = DSTEnv(load_dataset, model_arch, args)
    assert args.seed_size

    # Cowardly refusing to train
    # if env.load('seed'):
    #     raise ValueError("Cowardly refusing to train.")

    # Initialize evaluation
    best_metrics = env.metrics(True)
    for k, v in best_metrics.items():
        logger.log_metric(k, v, step=0)
    logging.info("Initial metrics: %s", best_metrics)

    # Train seed
    for epoch in range(1, args.epochs + 1):
        logging.info('Starting fit epoch %d.', epoch)
        env.fit()
        metrics = env.metrics(True)
        logging.info("Epoch metrics: %s", metrics)
        for k, v in metrics.items():
            logger.log_metric(k, v, step=epoch)
        if best_metrics is None or metrics[args.stop] > best_metrics[args.stop]:
            logging.info("Saving best!")
            env.save("seed")
            best_metrics = metrics


if __name__ == "__main__":
    main()
