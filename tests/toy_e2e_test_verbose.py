from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.toy.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.strategies.naive_baselines import epsilon_cheat
import pprint


def main():
    args = get_args()

    args.seed_size = 1
    args.label_budget = 2
    args.pool_size = 11
    args.al_batch = 2
    args.fit_items = 3
    args.batch_size = 2
    args.eval_period = 1
    args.recency_bias = 0
    args.seed = 911
    args.seed_epochs = 3
    args.epochs = 1

    env = DSTEnv(load_dataset, GLAD, args)
    print("Seed indices")
    print(env._support_ptrs)
    print("Stream indices")
    print(env._ptrs)
    print("\n")
    ended = False
    i = 0
    while not ended:
        i += 1
        print("Environment observation now.")
        raw_obs, obs_dist = env.observe()
        print("Current idx", env._current_idx)
        print("Current ptrs", env.current_ptrs)
        print("Raw observation:")
        pprint.pprint([d.to_dict() for d in raw_obs])
        print("Observation distribution:")
        pprint.pprint(obs_dist)
        true_labels = env.leak_labels()
        print("True labels:")
        pprint.pprint(true_labels)

        print("\n")
        requested_label = epsilon_cheat(obs_dist, true_labels)
        print("Requested label: ", requested_label)
        print("Environment label request now.")
        label_success = env.label(requested_label)
        print("Label success: ", label_success)
        print("Support ptrs: ", env._support_ptrs)
        print("Support mask: ", env._support_masks)
        print("Support labels: ", env._support_labels)

        print("\n")
        print("Environment stepping now.")
        ended = env.step()
        print("Ended: ", ended)

        print("\n")
        print("Fitting environment now.")
        env.fit()
        print("Reporting metrics now.")
        for k, v in env.metrics(i % args.eval_period == 0).items():
            print("\t{}: {}".format(k, v))

        print("\n")
        print("\n")
        print("\n")


if __name__ == "__main__":
    main()

