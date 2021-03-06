from comet_ml import Experiment
from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.models.gce import GCE
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key
from ActiveDialogue.strategies.vanilla_baselines import aggressive, random, passive
from ActiveDialogue.strategies.uncertainties import entropy, bald
from ActiveDialogue.strategies.common import FixedThresholdStrategy, VariableThresholdStrategy, StochasticVariableThresholdStrategy
import numpy as np
import sys
import logging


def main(cmd=None, stdout=True):
    args = get_args(cmd)

    model_id = "seed_{}_strat_{}_noise_fn_{}_noise_fp_{}_num_passes_{}_seed_size_{}_model_{}_batch_size_{}_gamma_{}_label_budget_{}_epochs_{}".format(
        args.seed, args.strategy, args.noise_fn, args.noise_fp, args.num_passes, args.seed_size, args.model, args.batch_size, args.gamma, args.label_budget, args.epochs)

    logging.basicConfig(
        filename="{}/{}.txt".format(args.dout, model_id),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    if stdout:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logger = Experiment(comet_ml_key, project_name="ActiveDialogue")
    logger.set_name("model {} strategy {}".format(args.model, args.strategy))
    logger.log_parameters(vars(args))

    if args.model == "glad":
        model_arch = GLAD
    elif args.model == "gce":
        model_arch = GCE

    env = DSTEnv(load_dataset, model_arch, args)
    if args.seed_size:
        with logger.train():
            if not env.load('seed'):
                logging.info("No loaded seed. Training now.")
                env.seed_fit(args.seed_epochs, prefix="seed")
                logging.info("Seed completed.")
            else:
                logging.info("Loaded seed.")
                if args.force_seed:
                    logging.info("Training seed regardless.")
                    env.seed_fit(args.seed_epochs, prefix="seed")
        env.load('seed')

    use_strategy = False
    if args.strategy == "entropy":
        use_strategy = True
        strategy = entropy
    elif args.strategy == "bald":
        use_strategy = True
        strategy = bald

    if use_strategy:
        if args.threshold_strategy == "fixed":
            strategy = FixedThresholdStrategy(strategy, args, False)
        elif args.threshold_strategy == "variable":
            strategy = VariableThresholdStrategy(strategy, args, False)
        elif args.threshold_strategy == "randomvariable":
            strategy = StochasticVariableThresholdStrategy(
                strategy, args, False)

    ended = False
    i = 0

    initial_metrics = env.metrics(True)
    logger.log_current_epoch(i)
    logging.info("Initial metrics: {}".format(initial_metrics))
    for k, v in initial_metrics.items():
        logger.log_metric(k, v)

    while not ended:
        i += 1

        # Observe environment state
        logger.log_current_epoch(i)

        if env.can_label:
            # Obtain label request from strategy
            obs, preds = env.observe(20 if args.strategy ==
                                     "bald" else 1)
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
                strategy.update(
                    np.sum(label_request.flatten()),
                    np.sum(np.ones_like(label_request.flatten())))
        else:
            break

        # Environment stepping
        ended = env.step()
        # Fit every al_batch of items
        best = env.fit(prefix=model_id, reset_model=False)
        for k, v in best.items():
            logger.log_metric(k, v)
        env.load(prefix=model_id)

    # Final fit
    final_metrics = env.fit(epochs=args.final_epochs,
                            prefix="final_fit_" + model_id,
                            reset_model=True)
    for k, v in final_metrics.items():
        logger.log_metric("Final " + k, v)
        logging.info("Final " + k + ": " + str(v))
    logging.info("Run finished.")


if __name__ == "__main__":
    main()
