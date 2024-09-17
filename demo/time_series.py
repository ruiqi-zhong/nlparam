from nlparam import run_time_series, DATA_DIR
import json
from typing import List, Dict


if __name__ == '__main__':

    # loading the inputs for time series
    problem_type, dataset_name = "time_series", "topics"
    dataset_path = DATA_DIR / problem_type / f"{dataset_name}.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    texts_by_time: List[str] = dataset['texts_by_time']

    K: int = dataset['K']
    goal: str = dataset['goal']

    time_series_result = run_time_series(texts_by_time, K, goal, dataset_name == "dummy")
    predicates: List[str] = time_series_result["predicates"]

    for i, predicate in enumerate(predicates):
        print(f"{predicate}: {time_series_result['curves'][i]}")
        matching_texts = [text for text, matching in time_series_result["predicate2text2matching"][predicate].items() if matching]
        print(f"Number of matching texts: {len(matching_texts)}")
        print(f"Example matching texts: {matching_texts[:5]}")
        curve = time_series_result["curves"][i]
        print(f"Curve: {curve}")

    