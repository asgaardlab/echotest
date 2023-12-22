from metaphone import doublemetaphone

from src.service.metaphone import compare_metaphone_approximations
from src.service.tokenization import tokenize

SHORT_WORD_CUTOFF_LENGTH = 5


class TextComparisonToken:
    """
    Helper class to compare words phonetically using metaphone approximations
    """

    def __init__(self, word: str):
        self.word = word
        self.tokenized = tokenize(word, segment=False, split_contractions=False)
        self.phonetic_representation = doublemetaphone(self.tokenized)

    def compare(self, other: 'TextComparisonToken'):
        """
        Compares one token to another to return a comparison score, cutoff at 0.5
        :param other: Comparison token to compare with
        :return: A score between 0.0 and 1.0
        """
        return compare_metaphone_approximations(self.phonetic_representation, other.phonetic_representation)

    def __repr__(self):
        return f"({self.word}, {self.phonetic_representation})"
