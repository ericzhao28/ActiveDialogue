from comet_ml import Experiment
from ActiveDialogue.environments.bag_env import BagEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key
from ActiveDialogue.strategies.confirmation_baselines import epsilon_cheat, random_singlets, passive
from ActiveDialogue.strategies.uncertainties import lc_singlet, bald_singlet
from ActiveDialogue.strategies.common import FixedThresholdStrategy, VariableThresholdStrategy, StochasticVariableThresholdStrategy


def main():
    args = get_args()
    logger = Experiment(comet_ml_key, project_name="ActiveDialogue")
    logger.log_parameters(vars(args))

    if args.model == "glad":
        model_arch = GLAD
    elif args.model == "gce":
        model_arch = GCE

    env = BagEnv(load_dataset, model_arch, args, logger)
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
        strategy = lc_singlet
    elif args.strategy == "bald":
        use_strategy = True
        strategy = bald_singlet

    if use_strategy:
        if args.threshold_strategy == "fixed":
            strategy = FixedThresholdStrategy(strategy, args)
        elif args.threshold_strategy == "variable":
            strategy = VariableThresholdStrategy(strategy, args)
        elif args.threshold_strategy == "randomvariable":
            strategy = StochasticVariableThresholdStrategy(strategy, args)

    ended = False
    i = 0
    with logger.train():
        while not ended:
            i += 1

            # Observe environment state
            logger.log_current_epoch(i)

            for j in range(args.label_timeout):
                if env.can_label:
                    # Obtain label request from strategy
                    obs, preds = env.observe()
                    if args.strategy == "epsiloncheat":
                        label_request = epsilon_cheat(preds, env.leak_labels())
                    elif args.strategy == "randomsinglets":
                        label_request = random_singlets(preds)
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
            env.fit()

    print("Final fit: ", env.seed_fit(100, "final_fit", True))


if __name__ == "__main__":
    main()
