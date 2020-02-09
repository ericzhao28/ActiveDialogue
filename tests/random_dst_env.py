from ActiveDialogue.main.utils import get_args
from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.woz.wrapper import load_dataset
from ActiveDialogue.strategies import naive_baselines
from ActiveDialogue.models.glad import GLAD


def main():
    args = get_args()
    env = DSTEnv(load_dataset, GLAD, args)
    ended = False
    while not ended:
        raw_obs, obs_dist = env.observe()
        env.label(naive_baselines.random_singlets(obs_dist))
        ended = env.step()
        env.fit()

    env.eval()


if __name__ == "__main__":
    main()

