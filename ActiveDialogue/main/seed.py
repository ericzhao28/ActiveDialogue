from comet_ml import Experiment
from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key, lib_dir
import sys
import logging


def main(args=None):
    if args is None:
        args = get_args()

    model_id = "seed_{}_seed_size_{}_model_{}_seed_batch_size_{}_seed_epochs_{}".format(
            args.seed, args.seed_size, args.model, args.seed_batch_size, args.seed_epochs)

    logging.basicConfig(
        filename=lib_dir + "/exp/" + model_id,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logger = Experiment(comet_ml_key, project_name="ActiveDialogue")
    logger.log_parameters(vars(args))

    if args.model == "glad":
        model_arch = GLAD
    elif args.model == "gce":
        model_arch = GCE

    env = DSTEnv(load_dataset, model_arch, args)
    assert args.seed_size

    with logger.train():
        if not env.load('seed'):
            logging.info("No loaded seed. Training now.")
            env.seed_fit(args.seed_epochs, prefix="seed", logger=logger)
            logging.info("Seed completed.")
        else:
            logging.info("Loaded seed.")
            if args.force_seed:
                logging.info("Training seed regardless.")
                env.seed_fit(args.seed_epochs, prefix="seed", logger=logger)
            else:
                logging.info("Not training seed---seed exists.")
        env.load('seed')

    initial_metrics = env.metrics(True)
    logging.info("Seed metrics: {}".format(initial_metrics))
    for k, v in initial_metrics.items():
        logger.log_metric("Final seed " + k, v)


if __name__ == "__main__":
    main()
