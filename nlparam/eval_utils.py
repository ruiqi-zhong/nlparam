import numpy as np
from typing import  List, Tuple, Dict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import scipy
from nlparam import query_wrapper
from nlparam import TEMPLATE_DIRECTORY
import os


def find_pair_wise_f_score(gold_tdm, predicted_tdm):
    gold_tdm, predicted_tdm = np.array(gold_tdm), np.array(predicted_tdm)

    fscore_matrix = np.zeros((gold_tdm.shape[1], predicted_tdm.shape[1]))

    for i in range(gold_tdm.shape[1]):
        for j in range(predicted_tdm.shape[1]):
            fscore_matrix[i, j] = f1_score(gold_tdm[:, i], predicted_tdm[:, j])

    return fscore_matrix


def compare_tdm(gold_tdm, predicted_tdm):
    gold_tdm, predicted_tdm = np.array(gold_tdm), np.array(predicted_tdm)

    fscore_matrix = find_pair_wise_f_score(gold_tdm, predicted_tdm)

    gold_matching_scores = fscore_matrix.max(axis=1)

    assert gold_matching_scores.shape[0] == gold_tdm.shape[1]

    return {
        "matched_f1_score": gold_matching_scores.mean(),
        "fscore_matrix": fscore_matrix,
        "matching_idxes": np.argmax(fscore_matrix, axis=1),
    }


def get_best_f1(y_true, y_pred):
    # find the percentile that maximizes f1 score
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    best_f1 = 0

    for percentile in np.linspace(0, 100, 10):
        threshold = np.percentile(y_pred, percentile)
        predicted_labels = y_pred >= threshold
        f1 = f1_score(y_true=y_true, y_pred=predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def normalize(scores):
    # scores shape: (num_texts, num_descriptions)
    scores = np.array(scores)
    scores = scores - np.mean(scores, axis=0)
    scores = scores / (np.linalg.norm(scores, axis=0) + 1e-8)
    return scores


def compute_pairwise_similarity(scores_1, scores_2):
    scores_1, scores_2 = normalize(scores_1), normalize(scores_2)
    pairwise_similarity = np.abs(np.matmul(scores_1.T, scores_2))
    return pairwise_similarity



def assign_labels(
    ground_truth_labels, predicted_labels, K
) -> Tuple[List[int], Dict[int, int]]:
    """
    Assigns predicted_labels to ground_truth_labels using the Hungarian algorithm.

    Parameters
    ----------
    ground_truth_labels: List[int]
        A list of ground truth labels.
    predicted_labels: List[int]
        A list of predicted labels.

    Returns
    -------
    Tuple[List[int], Dict[int, int]]
        A tuple of two elements. The first element is the best predicted labels for each ground truth label. The second element is a mapping from ground truth labels to predicted labels.
    """
    n = len(ground_truth_labels)
    assert n == len(predicted_labels)
    m = K
    mp = K
    cost_matrix = np.zeros((m, mp))
    for gt_label, pred_label in zip(ground_truth_labels, predicted_labels):
        cost_matrix[gt_label, pred_label] -= 1
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    row_ind, col_ind = row_ind.tolist(), col_ind.tolist()
    mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    for i in range(m):
        if i not in mapping:
            mapping[i] = np.argmax(cost_matrix[i, :]).item()
    return [mapping[p] for p in ground_truth_labels], mapping

def compare_text_description_matching(
    reference_matching: np.ndarray,
    predicted_matching: np.ndarray,
    hungarian_matching: bool
):
    """
    Compare the predicted matching to the reference matching.

    Parameters
    ----------
    reference_matching: np.ndarray
        A numpy array of shape (num_texts, num_descriptions) representing the reference matching.
    predicted_matching: np.ndarray
        A numpy array of shape (num_texts, num_descriptions) representing the predicted matching.

    Returns
    -------
    Dict[str, float]
        A dictionary of metrics.
    """

    if not hungarian_matching:
        result = compare_tdm(reference_matching, predicted_matching)
        result["main_metric"] = result["matched_f1_score"]
        return result

    assert reference_matching.shape == predicted_matching.shape, (
        f"reference_matching.shape={reference_matching.shape} "
        f"predicted_matching.shape={predicted_matching.shape}"
    )

    pairwise_similarity = compute_pairwise_similarity(
        scores_1=predicted_matching, scores_2=reference_matching
    )

    randomly_forced_pred_label = np.argmax(predicted_matching, axis=1)
    randomly_forced_ref_label = np.argmax(reference_matching, axis=1)

    N, K = predicted_matching.shape

    pred_label, mapping = assign_labels(
        randomly_forced_ref_label, randomly_forced_pred_label, K
    )

    matches = []
    for reference_idx in range(K):
        best_prediction_idx = mapping[reference_idx]
        match_dict = {
            "reference_idx": reference_idx,
            "best_prediction_idx": best_prediction_idx,
            "cosine_similarity": float(
                pairwise_similarity[best_prediction_idx, reference_idx]
            ),
            "auc_roc": roc_auc_score(
                y_true=reference_matching[:, reference_idx],
                y_score=predicted_matching[:, best_prediction_idx],
            ),
        }

        if np.allclose(
            predicted_matching[:, best_prediction_idx],
            predicted_matching[:, best_prediction_idx].astype(int),
        ):
            match_dict["f1_score"] = f1_score(
                y_true=reference_matching[:, reference_idx],
                y_pred=predicted_matching[:, best_prediction_idx],
            )
            match_dict["precision"] = precision_score(
                y_true=reference_matching[:, reference_idx],
                y_pred=predicted_matching[:, best_prediction_idx],
            )
            match_dict["recall"] = recall_score(
                y_true=reference_matching[:, reference_idx],
                y_pred=predicted_matching[:, best_prediction_idx],
            )
            match_dict["size"] = float(
                np.sum(predicted_matching[:, best_prediction_idx])
            )
        else:
            match_dict["f1_score"] = 0.0
            match_dict["precision"] = 0.0
            match_dict["recall"] = 0.0
            match_dict["size"] = 0.0
        matches.append(match_dict)

    result_dict = {"matching_by_reference_description": matches}
    for k in ["cosine_similarity", "auc_roc", "f1_score"]:
        result_dict["description_mean_" + k] = np.mean([m[k] for m in matches])
    result_dict["main_metric"] = result_dict["description_mean_f1_score"]
    result_dict["matching_idxes"] = [mapping[i] for i in range(K)]
    return result_dict


SIMILARITY_TEMPLATE_PATH = os.path.join(TEMPLATE_DIRECTORY, "similarity_judgment.txt")
with open(SIMILARITY_TEMPLATE_PATH) as f:
    SIMILARITY_TEMPLATE = f.read()

def parse_similarity_score(s):
    if type(s) != str:
        return 0
    s = s.strip()
    if "yes" in s.lower():
        return 1
    if "no" in s.lower():
        return 0
    return 0.5


def is_null_description(s):
    return len(str(s).strip().split(" ")) <= 1


def get_similarity_scores(
    texts_a: List[str], texts_b: List[str], model: str = "gpt-4o"
) -> List[int]:
    assert len(texts_a) == len(texts_b)

    results = [1 if text_a == text_b else 0 for text_a, text_b in zip(texts_a, texts_b)]

    need_to_query_idxes = [
        i
        for i in range(len(texts_a))
        if not is_null_description(texts_a[i]) and not is_null_description(texts_b[i])
    ]
    if len(need_to_query_idxes) == 0:
        return results

    prompts = [
        SIMILARITY_TEMPLATE.format(text_a=texts_a[i], text_b=texts_b[i])
        for i in need_to_query_idxes
    ]
    responses = query_wrapper(
        prompts=prompts,
        model=model,
        num_processes=5,
        temperature=0.0,
        progress_bar=True,
    )
    ratings = [parse_similarity_score(r) for r in responses]
    for i, r in zip(need_to_query_idxes, ratings):
        results[i] = r
    return results

def extract_performance_from_opt_traj(opt_traj, commit, iter_idx):
    result_dict = opt_traj[iter_idx]
    learned_predicates = result_dict["predicate_strings"]
    commit_key = "after_commitment" if commit else "before_commitment"
    main_f1_metric = result_dict["eval_phi_predicate_strings"][commit_key]["main_metric"]
    surface_similarity_metric = result_dict["eval_phi_predicate_strings"]["surface_similarity_scores"]
    denotation = result_dict["eval_phi_predicate_strings"][commit_key + "_denotation"]
    return {
        "learned_predicates": learned_predicates,
        "main_f1_metric": main_f1_metric,
        "surface_similarity_metric": surface_similarity_metric,
        "denotation": denotation,
    }