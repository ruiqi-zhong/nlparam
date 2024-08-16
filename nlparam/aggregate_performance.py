import pickle as pkl
import numpy as np
import os
import pandas as pd
from nlparam.eval_utils import extract_performance_from_opt_traj

if __name__ == '__main__':
    result_dir = "results/"
    problem_name2tag2performances = {}
    metric_name = "main_f1_metric"
    for f in os.listdir(result_dir):
        if f.endswith(".pkl"):
            with open(result_dir + f, "rb") as file:
                data = pkl.load(file)
                if type(data) != dict:
                    continue
                
                problem_type = data["problem_type"]
                dataset_name = data["dataset_name"]
                if dataset_name == "dummy":
                    continue
                f1_metric = data["final_performance"][metric_name]
                tag = "normal"
                if data["random_update"]:
                    tag = "random_update"
                if data["one_hot_embedding"]:
                    tag = "one_hot"

                problem_name = f"{problem_type}_{dataset_name}"
                if problem_name not in problem_name2tag2performances:
                    problem_name2tag2performances[problem_name] = {}
                if tag not in problem_name2tag2performances[problem_name]:
                    problem_name2tag2performances[problem_name][tag] = []
                
                problem_name2tag2performances[problem_name][tag].append(f1_metric)

                if tag == "normal":
                    first_round_performance = extract_performance_from_opt_traj(data["opt_traj"], problem_type == "cluster", 0)
                    first_round_tag = "no-refine"
                    if first_round_tag not in problem_name2tag2performances[problem_name]:
                        problem_name2tag2performances[problem_name][first_round_tag] = []
                    
                    problem_name2tag2performances[problem_name][first_round_tag].append(first_round_performance[metric_name])
    
    for problem_name, tag2performances in problem_name2tag2performances.items():
        for tag in tag2performances:
            performances = tag2performances[tag]
            mean_performance = np.mean(performances)
            tag2performances[tag] = mean_performance
    
    # reorder the columns to be normal, random_update, one_hot
    df = pd.DataFrame(problem_name2tag2performances).T
    columns = ["normal", "random_update", "one_hot", "no-refine"]
    df = df[columns]

    problem_types = set([problem_name.split("_")[0] for problem_name in df.index])
    for problem_type in problem_types:
        problem_names = [problem_name for problem_name in df.index if problem_name.startswith(problem_type)]
        sub_df = df.loc[problem_names]
        # get the avg for each column
        sub_df.loc["avg"] = sub_df.mean()
        print(sub_df)
