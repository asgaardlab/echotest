from typing import Tuple

import Levenshtein


def compare_metaphone_approximations(representation_a: Tuple[str, str], representation_b: Tuple[str, str]) -> float:
    """
    Gives a similarity score for two metaphone approximated representations
    """
    if any_match(representation_a, representation_b):
        return 1.0   # Perfect metaphone representation match
    similarity_score = 0.0
    for i in range(0, 2):
        for j in range(0, 2):
            representation_length = max(len(representation_a[i]), len(representation_b[j]))
            a = representation_a[i].ljust(representation_length)
            b = representation_b[j].ljust(representation_length)
            if not a or not b:
                continue
            similarity_score = max(similarity_score, Levenshtein.ratio(a, b, score_cutoff=0.51))
    return similarity_score


def any_match(representation_a, representation_b):
    """
    Returns true if there is any match between the tuples in representation A and representation B.
    Only considers non empty representation
    """
    return any([(representation == representation_a[0] and representation_a[0])
                or (representation == representation_a[1] and representation_a[1]) for
                representation in representation_b])
