import re

import contractions
import nltk
import wordninja


def tokens_equal(a: str, b: str) -> bool:
    """
    Checks token equality between two strings
    """
    if not a or not b:
        return False
    return tokenize(a) == tokenize(b)


def tokenize(text: str, segment=True, split_contractions=True) -> str:
    """
    Performs case-folding and punctuation removal, while optionally segmenting and splitting contractions
    """
    pattern = r"\b\w+(?:['’]\w+)?\b"
    tokenizer = nltk.RegexpTokenizer(pattern)
    if segment:
        text = perform_segmentation(text)
    if split_contractions:
        text = contractions.fix(text)
    tokenized = tokenizer.tokenize(text)
    return " ".join(tokenized).lower()


def perform_segmentation(text: str) -> str:
    """
    Performs text segmentation
    """
    return " ".join([" ".join(wordninja.split(word)) if not word.istitle() else word for word in text.split()])


def segment(text: str) -> str:
    """
    Performs text segmentation while maintaining punctuation
    """
    punctuated_words = re.findall(r"\w+[.,:;]", text)
    segmented_words = [" ".join(wordninja.split(word))
                       if (not word.istitle() and word not in punctuated_words) else word for word in text.split()]
    return " ".join(segmented_words)


def is_contraction(word: str) -> bool:
    """
    Returns true if a given word represents a common contraction
    """
    return contraction_length(word) > 1


def contraction_length(word: str) -> int:
    """
    Returns the length of a contraction. For example "Shouldn't've" -> "Should not have" -> 3
    If a word does not represent a contraction then the length that will be returned is 1
    """
    return len(contractions.fix(word).split())


def segmentation_length(word: str) -> int:
    """
    Returns the length of a segmented word. For example "totime" -> "to time" -> 2
    """
    return len(wordninja.split(word))
