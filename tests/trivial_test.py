from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.toy.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.strategies.naive_baselines import epsilon_cheat
import pprint


def main():
    args = get_args()

    env = DSTEnv(load_dataset, GLAD, args)
    ended = False
    i = 0
    while not ended:
        i += 1
        print("Environment observation now.")
        raw_obs, obs_dist = env.observe()
        print("Raw observation:")
        pprint.pprint(raw_obs)
        print("Observation distribution:")
        pprint.pprint(obs_dist)
        true_labels = env.leak_labels()
        print("True labels:")
        pprint.pprint(true_labels)
        requested_label = epsilon_cheat(obs_dist, true_labels)
        print("Environment label request now.")
        label_success = env.label(requested_label)
        print("Label success: ", label_success)
        print("Environment stepping now.")
        ended = env.step()
        print("Ended: ", ended)
        print("Fitting environment now.")
        env.fit()
        print("Reporting metrics now.")
        for k, v in env.metrics(i % args.eval_period == 0).items():
            print("\t{}: {}".format(k, v))


if __name__ == "__main__":
    main()

