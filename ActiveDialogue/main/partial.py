from comet_ml import Experiment
import numpy as np
from ActiveDialogue.environments.partial_env import PartialEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key
from ActiveDialogue.strategies.partial_baselines import aggressive, random, passive


def main():
    args = get_args()
    logger = Experiment(comet_ml_key, project_name="ActiveDialogue")
    logger.log_parameters(vars(args))

    if args.model == "glad":
        model_arch = GLAD
    elif args.model == "gce":
        model_arch = GCE

    env = PartialEnv(load_dataset, model_arch, args)
    if args.seed_size:
        if not env.load_seed():
            print("No loaded seed. Training now.")
            env.fit(args.seed_epochs, prefix="seed")
            print("Seed completed.")
        else:
            print("Loaded seed.")
            if args.force_seed:
                print("Training seed regardless.")
                env.fit(args.seed_epochs, prefix="seed")
        print("Current seed metrics:", env.metrics(True))

    ended = False
    can_label = True

    i = 0
    with logger.train():
        while not ended:
            i += 1

            # Observe environment state
            logger.log_current_epoch(i)
            obs, preds = env.observe()

            if can_label:
                # Obtain label request from strategy
                if args.strategy == "aggressive":
                    label_request = aggressive(preds)
                elif args.strategy == "random":
                    label_request = random(preds)
                elif args.strategy == "passive":
                    label_request = passive(preds)
                else:
                    raise ValueError()

                # Label solicitation
                label_occured = env.confirmation_label(label_request)

            # Environment stepping
            ended = env.step()
            # Fit every al_batch of items
            env.fit()

            # Report metrics to stdout and comet
            for k, v in env.metrics(i % args.eval_period == 0).items():
                print(k, v)
                logger.log_metric(k, v, step=i)


if __name__ == "__main__":
    main()

