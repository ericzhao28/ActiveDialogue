# Train model

You can checkout the training options via `python train.py -h`.
By default, `train.py` will save checkpoints to `exp/glad/default`.

```
docker exec glad python train.py --gpu 0
```

You can attach to the container via `docker exec glad -it bin/bash` to look at what's inside or `docker cp glad /opt/glad/exp exp` to copy out the experiment results.

If you do not want to build the Docker image, then run

```
python train.py --gpu 0
```


# Evaluation

You can evaluate the model using

```
docker exec glad python evaluate.py --gpu 0 --split test exp/glad/default
```

You can also dump a predictions file by specifying the `--fout` flag.
In this case, the output will be a list of lists.
Each `i`th sublist is the set of predicted slot-value pairs for the `i`th turn.
Please see `evaluate.py` to see how to match up the turn predictions with the dialogues.

If you do not want to build the Docker image, then run

```
python evaluate.py --gpu 0 --split test exp/glad/default
```


