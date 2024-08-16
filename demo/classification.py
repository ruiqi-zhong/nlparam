from nlparam import ClassificationModel, ModelOutput
import json
from sklearn.linear_model import LogisticRegression
from typing import List, Dict


if __name__ == '__main__':
    data_path = 'applications/demo_mem.json'
    with open(data_path, "r") as f:
        task_dict = json.load(f)

    texts: List[str] = task_dict["texts"]
    labels: List[int] = task_dict["labels"]
    K: int = 3
    goal: str = "Here are some captions of the images. I am a cognitive scientist and I want to understand what visual features are important for people to remember an image."

    model = ClassificationModel(
        texts=texts,
        labels=labels,
        K=K,
        goal=goal,
    )
    result: ModelOutput = model.fit()
    predicate2text2matching: Dict[str, Dict[str, int]] = result.predicate2text2matching

    predicates = list(predicate2text2matching.keys())
    features = [
        [predicate2text2matching[predicate][text] for predicate in predicates]
        for text in texts
    ]
    clf = LogisticRegression()
    clf.fit(features, labels)
    w = clf.coef_[0]

    for i, predicate in enumerate(predicates):
        print(f"{predicate}: {w[i]}")

    