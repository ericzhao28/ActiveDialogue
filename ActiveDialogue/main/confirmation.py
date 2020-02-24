from comet_ml import Experiment
import numpy as np
from ActiveDialogue.environments.bag_env import BagEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key
from ActiveDialogue.strategies.naive_baselines import epsilon_cheat, random_singlets, passive


def main():
    args = get_args()
    logger = Experiment(comet_ml_key, project_name="ActiveDialogue")
    logger.log_parameters(vars(args))

    if args.model == "glad":
        model_arch = GLAD
    elif args.model == "gce":
        model_arch = GCE

    env = BagEnv(load_dataset, model_arch, args)
    ended = False
    can_label = True

    i = 0
    with logger.train():
        while not ended:
            i += 1

            # Observe environment state
            logger.log_current_epoch(i)
            raw_obs, obs_dist = env.observe()

            if can_label:
                for j in range(args.label_timeout):
                    # Obtain label request from strategy
                    if args.strategy == "epsiloncheat":
                        label_request = epsilon_cheat(obs_dist,
                                                      env.leak_labels())
                    elif args.strategy == "randomsinglets":
                        label_request = random_singlets(obs_dist)
                    elif args.strategy == "passive":
                        label_request = passive(obs_dist)
                    else:
                        raise ValueError()

                    # Check if request is trivial
                    request_empty = True
                    for v in label_request.values():
                        if np.any(v):
                            request_empty = False
                            break
                    if args.strategy == "passive":
                        assert request_empty
                    if request_empty:
                        break

                    # Label solicitation
                    label_occured = env.confirmation_label(label_request)

                    # At this point, label request is non trivial but no
                    # labeling occured, so we assume budget is exhausted.
                    if not label_occured:
                        can_label = False

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
