import math
import random

import nltk

from src.service.tokenization import segment

FRACTION_MODIFIED = 0.1
FRACTION_REMOVED = 0.25

MAX_QUOTE_LEN = 250
MAX_SUBTITLE_LEN_CHARS = 60

nltk.download('words')
from nltk.corpus import words

quotes_tsv = "data/quotes.tsv"
words = words.words()


def generate_phrases(num_phrases: int):
    """
    Selects a number of phrases from a database of phrases and then modifies them to add errors and word modifications
    """
    quotes = open(quotes_tsv, 'r').readlines()
    total_phrases = len(quotes)
    modified_phrases = []
    for _ in range(0, num_phrases):
        quote = None
        while not quote or len(quote) > MAX_QUOTE_LEN:
            quote = quotes[random.randint(0, total_phrases)]
            try:
                quote.encode()
            except UnicodeEncodeError:
                quote = None
                continue
        segmented = segment(quote)
        broken_up = break_up_phrase(segmented)

        modified_phrase = []
        for section in broken_up:
            modified_phrase.append(permutate_phrase(section.strip(),
                                                    random.uniform(0, FRACTION_MODIFIED),
                                                    random.uniform(0, FRACTION_REMOVED)))
        modified_phrases.append(modified_phrase)
    return modified_phrases


def break_up_phrase(phrase: str):
    """
    Splits a phrase into multiple sections in order to avoid subtitle overflow
    """
    broken_down_phrase = []
    if len(phrase) > MAX_SUBTITLE_LEN_CHARS:
        sentences = split_keep_delimeter(phrase, ".")
        for sentence in sentences:
            if len(sentence) > MAX_SUBTITLE_LEN_CHARS:
                split_sentences = split_keep_delimeter(sentence, ",")
                broken_down_phrase.extend(split_sentences)
            else:
                broken_down_phrase.append(sentence)
    else:
        broken_down_phrase = [phrase]
    return broken_down_phrase


def split_keep_delimeter(sentence: str, delimiter: str):
    splits = [(split + delimiter).strip() for split in sentence.split(delimiter)]
    if splits[-1] == delimiter:
        splits.pop()
    else:
        splits[-1] = splits[-1].replace(delimiter, "")
    return splits


def permutate_phrase(phrase: str, fraction_modified: float, fraction_missed: float):
    """
    Changes num_permutation words in the phrase to similar words and num_mistake
    words in the phrase to completely dissimilar words

    :param phrase: to permutate
    :param fraction_modified: used to calculate the number of words that are altered,
    words are altered to have the same first bi-gram (words that are two letters or less are skipped)
    :param fraction_missed: used to calculate the number of words that are completely removed from the permutation

    :returns: Returns the changed phrase in a list which represents which words are altered
    and which words are removed
    """
    if 0 > fraction_missed > 1 or 0 > fraction_modified > 1:
        raise ValueError("Percent modified and missed must be numbers between 0 and 1")
    phrase_statistics = []
    for word in phrase.split():
        phrase_statistics.append({"word": word, "removed": False, "modified": False, "final_word": word})
    num_words_to_remove = math.floor(len(phrase.split()) * fraction_missed)
    num_words_to_modify = math.floor(len(phrase.split()) * fraction_modified)

    # Remove random words from the final phrase and update the statistics
    for i in range(0, num_words_to_remove):
        non_removed_words = [word for word in phrase_statistics if not word["removed"]]
        word_to_remove = find_random(non_removed_words)
        if not word_to_remove:
            break
        word_statistics = phrase_statistics[phrase_statistics.index(word_to_remove)]
        word_statistics["removed"] = True
        word_statistics["final_word"] = ""

    # Permutate non-removed words in the final phrase and update the statistics
    non_removed_words = [word for word in phrase_statistics if not word["removed"]]
    for i in range(0, num_words_to_modify):
        non_removed_non_modified = [word for word in non_removed_words if not word["modified"] and len(word) > 2]
        word_to_modify = find_random(non_removed_non_modified)
        if not word_to_modify:
            break
        word_statistics = phrase_statistics[phrase_statistics.index(word_to_modify)]
        word_statistics["modified"] = True
        word_statistics["final_word"] = find_word_with_same_bigram(word_to_modify["word"])

    return phrase_statistics


def find_random(target_list: list):
    """
    Returns a random element from a given target_list or none if the list is empty
    """
    if not len(target_list):
        return None
    element = target_list[random.randint(0, len(target_list) - 1)]
    return element


def find_word_with_same_bigram(reference_word):
    """
    Finds another valid English word which shares the first bigram with the reference word
    :param reference_word: word to permute
    """
    first_bigram = reference_word[:2]
    valid_words = [word for word in words if word != reference_word
                   and word.startswith(first_bigram)
                   and 2 < len(word) < len(reference_word) + 2]
    return find_random(valid_words)


if __name__ == '__main__':
    stats = generate_phrases(100)
