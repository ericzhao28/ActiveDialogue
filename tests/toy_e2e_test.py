from ActiveDialogue.environments.dst_env import DSTEnv
from ActiveDialogue.datasets.toy.wrapper import load_dataset
from ActiveDialogue.models.glad import GLAD
from ActiveDialogue.main.utils import get_args
from ActiveDialogue.strategies.naive_baselines import epsilon_cheat
import pprint


def main():
    args = get_args()

    args.seed_size = 0
    args.label_budget = 1000
    args.pool_size = 1000
    args.al_batch = 20
    args.fit_items = 3
    args.batch_size = 2
    args.comp_batch_size = 2
    args.eval_period = 1
    args.recency_bias = 0
    args.seed = 912
    args.seed_epochs = 40
    args.epochs = 1

    env = DSTEnv(load_dataset, GLAD, args)
    ended = False
    i = 0
    while not ended:
        i += 1
        raw_obs, obs_dist = env.observe()
        true_labels = env.leak_labels()

        requested_label = epsilon_cheat(obs_dist, true_labels)
        label_success = env.label(requested_label)
        ended = env.step()

        env.fit()
        for k, v in env.metrics(i % args.eval_period == 0).items():
            print("\t{}: {}".format(k, v))

    print(env._support_masks)
    print(env._support_labels)
    for v in env._support_masks.values():
        assert np.all(v)


if __name__ == "__main__":
    main()

