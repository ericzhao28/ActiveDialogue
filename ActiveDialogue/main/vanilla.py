from comet_ml import Experiment
from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key
from ActiveDialogue.strategies.vanilla_baselines import aggressive, random, passive
from ActiveDialogue.strategies.uncertainties import lc, bald
from ActiveDialogue.strategies.common import FixedThresholdStrategy, VariableThresholdStrategy, StochasticVariableThresholdStrategy
import numpy as np


def main():
    args = get_args()
    logger = Experiment(comet_ml_key, project_name="ActiveDialogue")
    logger.log_parameters(vars(args))

    if args.model == "glad":
        model_arch = GLAD
    elif args.model == "gce":
        model_arch = GCE

    env = DSTEnv(load_dataset, model_arch, args, logger)
    if args.seed_size:
        with logger.train():
            if not env.load_seed():
                print("No loaded seed. Training now.")
                env.seed_fit(args.seed_epochs, prefix="seed")
                print("Seed completed.")
            else:
                print("Loaded seed.")
                if args.force_seed:
                    print("Training seed regardless.")
                    env.seed_fit(args.seed_epochs, prefix="seed")
        env.load_seed()
        print("Current seed metrics:", env.metrics(True))

    use_strategy = False
    if args.strategy == "lc":
        use_strategy = True
        strategy = lc
    elif args.strategy == "bald":
        use_strategy = True
        strategy = bald

    if use_strategy:
        if args.threshold_strategy == "fixed":
            strategy = FixedThresholdStrategy(strategy, args, False)
        elif args.threshold_strategy == "variable":
            strategy = VariableThresholdStrategy(strategy, args, False)
        elif args.threshold_strategy == "randomvariable":
            strategy = StochasticVariableThresholdStrategy(strategy, args, False)

    ended = False
    i = 0
    with logger.train():
        while not ended:
            i += 1

            # Observe environment state
            logger.log_current_epoch(i)

            if env.can_label:
                # Obtain label request from strategy
                obs, preds = env.observe(100 if args.strategy == "bald" else 1)
                if args.strategy != "bald":
                    preds = preds[0]
                if args.strategy == "aggressive":
                    label_request = aggressive(preds)
                elif args.strategy == "random":
                    label_request = random(preds)
                elif args.strategy == "passive":
                    label_request = passive(preds)
                elif use_strategy:
                    label_request = strategy.observe(preds)
                else:
                    raise ValueError()

                # Label solicitation
                labeled = env.label(label_request)
                if use_strategy:
                    strategy.update(np.sum(label_request.flatten()),
                                    np.sum(np.ones_like(label_request.flatten()))
                                    )

            # Environment stepping
            ended = env.step()
            # Fit every al_batch of items
            env.fit(prefix=env.id())

    # Final fit
    print("Final fit: ", env.fit(epochs=20, prefix="final_fit_" + env.id(), reset_model=True))


if __name__ == "__main__":
    main()
