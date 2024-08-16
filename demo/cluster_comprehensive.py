from nlparam import ClusteringModel, DATA_DIR, ModelOutput
import json

if __name__ == '__main__':

    # loading the corpus
    with_reference = True
    problem_type, dataset_name = "cluster", "agnews"
    dataset_path = DATA_DIR / problem_type / f"{dataset_name}.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    texts, K, goal = dataset['texts'], dataset['K'], dataset['goal']
    print(f"Number of text samples: {len(texts)}")
    print(f"Example text samples from the dataset to be clustered:", texts[:5])
    print(f"Number of clusters: {K}")
    print(f"Clustering goal: {goal}")

    # gold reference
    reference_phi_predicate_strings = dataset["reference_phi_predicate_strings"]
    print(f"Reference predicates: {reference_phi_predicate_strings}")
    if not with_reference:
        reference_phi_predicate_strings = None
    
    reference_phi_denotation = dataset["reference_phi_denotation"]
    print(f"Reference denotation of the first few examples. This should be a N x K binary matrix: {reference_phi_denotation[:5]}")
    print(f"Referece_phi_denotation.shape[i][j] = 1 if the i-th text matches the j-th predicate")
    if not with_reference:
        reference_phi_denotation = None
    
    texts = dataset['texts']
    print(f"Number of text samples: {len(texts)}")
    print(f"Example text samples from the dataset to be clustered:")
    print(texts[:5])

    K = dataset['K']
    print(f"Number of clusters: {K}")

    # the goal of clustering
    goal = dataset['goal']
    print(f"Clustering goal: {goal}")

    model = ClusteringModel(
        texts=texts,
        K=K,
        goal=goal,
        dummy=dataset_name == "dummy",
        reference_phi_predicate_strings=reference_phi_predicate_strings,
        reference_phi_denotation=reference_phi_denotation,
    )
    result: ModelOutput = model.fit()

    predicate2text2matching = result.predicate2text2matching

    print(f"Learned predicates: {list(predicate2text2matching.keys())}")
    print(f"Reference predicates: {reference_phi_predicate_strings}")

    print(f"Performance evaluated by F1 similarity: {result.f1_similarity_score}")

    for predicate, text2matching in predicate2text2matching.items():
        print(f"Predicate: {predicate}")
        matching_texts = [text for text, matching in text2matching.items() if matching]
        print(f"Number of matching texts: {len(matching_texts)}")
        print(f"Example matching texts: {matching_texts[:5]}")