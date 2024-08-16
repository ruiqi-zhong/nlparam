from nlparam import TimeSeriesModel, ModelOutput, DATA_DIR, continuous_arr
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

    # model fitting
    model = TimeSeriesModel(
        texts_by_time=texts_by_time,
        K=K,
        goal=goal,
        dummy=dataset_name == "dummy",
    )
    model_output: ModelOutput = model.fit()

    # displaying the results
    learned_predicates: List[str] = list(model_output.predicate2text2matching.keys())
    predicate2text2matching: Dict[str, Dict[str, int]] = model_output.predicate2text2matching

    for predicate, text2matching in predicate2text2matching.items():
        print(f"Predicate: {predicate}")
        matching_texts = [text for text, matching in text2matching.items() if matching]
        print(f"Number of matching texts: {len(matching_texts)}")
        print(f"Example matching texts: {matching_texts[:5]}")
        features_by_time = [predicate2text2matching[predicate][text] for text in texts_by_time]
        curve = continuous_arr(features_by_time)
        # the curve of how much text matches the predicate over time
        print(f"Curve: {curve}")

    