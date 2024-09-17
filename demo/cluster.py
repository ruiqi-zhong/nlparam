from nlparam import run_clustering
import json
from typing import List, Dict

if __name__ == '__main__':
    # loading the inputs for clustering
    data_path = "applications/demo_math.json"
    with open(data_path, "r") as f:
        texts: List[str] = json.load(f)
    K: int = 5
    goal: str = "I want to cluster these math problems based on the type of skills required to solve them."

    clustering_result = run_clustering(texts, K, goal)
    predicate2text2matching: Dict[str, Dict[str, int]] = clustering_result["predicate2text2matching"]

    for predicate, text2matching in predicate2text2matching.items():
        print(f"Predicate: {predicate}")
        matching_texts = [text for text, matching in text2matching.items() if matching]
        print(f"Number of matching texts: {len(matching_texts)}")
        print(f"Example matching texts: {matching_texts[:5]}")