from ActiveDialogue.config import comet_ml_key
import comet_ml
from csv import DictWriter


comet_api = comet_ml.api.API(rest_api_key=comet_ml_key)

# LC partial
data = comet_api.get_experiment_by_id("ba94c26324fe429ab5780277487a8905").get_metrics()
with open("lc.csv", "w") as f:
    w = DictWriter(f, data[0].keys())
    w.writeheader()
    for d in data:
        w.writerow(d)

# Aggressive partial
data = comet_api.get_experiment_by_id("65e64e3172004ac5bbc33320ee2fdcc0").get_metrics()
with open("agg.csv", "w") as f:
    w = DictWriter(f, data[0].keys())
    w.writeheader()
    for d in data:
        w.writerow(d)
