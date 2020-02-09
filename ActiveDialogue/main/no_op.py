from comet_ml import Experiment
from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.config import comet_ml_key


def main():
    args = get_args()
    logger = Experiment(comet_ml_key, project_name="UNI-PPLM")
    logger.log_parameters(vars(args))

    env = DSTEnv(load_dataset, GLAD, args)
    ended = False
    i = 0
    with logger.train():
        while not ended:
            i += 1
            logger.log_current_epoch(i)
            raw_obs, obs_dist = env.observe()
            ended = env.step()
            env.fit()
            for k, v in env.metrics(i % args.eval_period == 1).items():
                logger.log_metric(k, v, step=i)


if __name__ == "__main__":
    main()
