import pickle as pkl
from nlparam import Proposer, get_validator_by_name, Embedder, DEFAULT_EMBEDDER_NAME, logger, EXPERIMENT_VALIDATOR_NAME, EXPERIMENT_PROPOSER_NAME, DATA_DIR
import json
from nlparam.models.classification_model import ClassificationModel
from nlparam.models.cluster_model import ClusteringModel
from nlparam.models.timeseries_model import TimeSeriesModel
import time
import random
import torch
import numpy as np
import os
from argparse import ArgumentParser
from nlparam.eval_utils import extract_performance_from_opt_traj

problem_type2class = {
    "cluster": ClusteringModel,
    "classification": ClassificationModel,
    "time_series": TimeSeriesModel,
}

problem_type2dataset_names = {
    problem_type: [
        f.split(".json")[0] for f in os.listdir(DATA_DIR / problem_type) if f.endswith(".json")
    ]
    for problem_type in problem_type2class
}


def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_experiment(problem_type, dataset_name, seed, one_hot_embedding=False, random_update=False):
    
    init_seed(seed)
    current_time = time.time()
    
    if not one_hot_embedding:
        if not random_update:
            save_path = f"results/{problem_type}_{dataset_name}_{seed}.pkl"
        else:
            save_path = (
                f"results/{problem_type}_{dataset_name}_{seed}_random_update.pkl"
            )
    else:
        save_path = f"results/{problem_type}_{dataset_name}_{seed}_one_hot.pkl"

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            result = pkl.load(f)
        if type(result) != float or current_time - result < 60 * 60 * 24:
            logger.debug(f"Skipping {problem_type} on {dataset_name} with seed {seed}")
            return result
    else:
        with open(save_path, "wb") as f:
            pkl.dump(current_time, f)

    dataset_path = DATA_DIR / problem_type / f"{dataset_name}.json"

    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    logger.debug(
        f"Running experiment for {problem_type} on {dataset_name} with seed {seed}"
    )

    if dataset_name == "dummy":
        embedder, validator, proposer = Embedder("dummy" if not one_hot_embedding else "one_hot"), get_validator_by_name("dummy"), Proposer("dummy")
    else:
        embedder, validator, proposer = Embedder(DEFAULT_EMBEDDER_NAME if not one_hot_embedding else "one_hot"), get_validator_by_name(EXPERIMENT_VALIDATOR_NAME), Proposer(EXPERIMENT_PROPOSER_NAME, simple_predicate=True),
    
    model_class = problem_type2class[problem_type]
    model = model_class(
        **dataset,
        embedder=embedder,
        validator=validator,
        proposer=proposer,
        random_update=random_update,
    )
    opt_traj = model.full_optimization_loop()
    final_performance = extract_performance_from_opt_traj(opt_traj, commit=model.commit, iter_idx=-1)
    final_performance["texts"] = model.texts

    dumped_dict = {
        "final_performance": final_performance,
        "opt_traj": opt_traj,
        "seed": seed,
        "problem_type": problem_type,
        "dataset_name": dataset_name,
        "one_hot_embedding": one_hot_embedding,
        "random_update": random_update,
    }

    with open(save_path, "wb") as f:
        pkl.dump(dumped_dict, f)

    return dumped_dict


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--one_hot_embedding", action="store_true", default=False)
    parser.add_argument("--random_update", action="store_true", default=False)
    parser.add_argument("--run_dummy", action="store_true", default=False)
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    one_hot_embedding = args.one_hot_embedding
    random_update = args.random_update
    assert not (random_update and one_hot_embedding), "does not support random_update and one_hot_embedding experimentation at the same time"
    run_dummy = args.run_dummy

    
    if not run_dummy:
        for seed in range(args.num_seeds):
            for problem_type in problem_type2class:
                for dataset_name in problem_type2dataset_names[problem_type]:
                    if "dataset" == "dummy":
                        continue
                    run_experiment(problem_type, dataset_name, seed, one_hot_embedding=one_hot_embedding, random_update=random_update)
    else:
        for seed in range(args.num_seeds):
            for problem_type in problem_type2class:
                result = run_experiment(problem_type, "dummy", seed, one_hot_embedding=one_hot_embedding, random_update=random_update)
                performance = result["final_performance"]["main_f1_metric"]
                print(f"Performance for {problem_type} dummy dataset: {performance}")
