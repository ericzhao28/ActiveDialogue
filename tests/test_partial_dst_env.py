from ActiveDialogue.environments.partial_env import PartialEnv
from ActiveDialogue.datasets.toy.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.strategies.partial_baselines import aggressive
import logging


def main():
    args = get_args()

    args.seed_size = 1
    args.label_budget = 2
    args.num_passes = 1
    args.al_batch = 2
    args.fit_items = 3
    args.batch_size = 2
    args.comp_batch_size = 2
    args.eval_period = 1
    args.recency_bias = 0
    args.seed = 911
    args.seed_epochs = 3
    args.epochs = 1

    env = PartialEnv(load_dataset, GLAD, args)
    logging.info("Seed indices")
    logging.info(env._support_ptrs)
    logging.info("Stream indices")
    logging.info(env._ptrs)
    logging.info("\n")
    ended = False
    i = 0
    while not ended:
        i += 1
        logging.info("Environment observation now.")
        raw_obs, preds = env.observe(1)
        pred = preds[0]
        logging.info("Current idx", env._current_idx)
        logging.info("Current ptrs", env.current_ptrs)
        logging.info("Raw observation:")
        logging.info([d.to_dict() for d in raw_obs])
        logging.info("Observation distribution:")
        logging.info(pred)
        true_labels = env.leak_labels()
        logging.info("True labels:")
        logging.info(true_labels)

        logging.info("\n")
        requested_label = aggressive(pred)
        logging.info("Requested label: ", requested_label)
        logging.info("Environment label request now.")

        if env.can_label:
            label_success = env.label(requested_label)
            logging.info("Label success: ", label_success)
            logging.info("Support ptrs: ", env._support_ptrs)

        logging.info("\n")
        logging.info("Environment stepping now.")
        ended = env.step()
        logging.info("Ended: ", ended)

        logging.info("\n")
        logging.info("Fitting environment now.")
        env.fit()
        logging.info("Reporting metrics now.")
        for k, v in env.metrics(i % args.eval_period == 0).items():
            logging.info("\t{}: {}".format(k, v))

        logging.info("\n")
        logging.info("\n")
        logging.info("\n")


if __name__ == "__main__":
    main()

