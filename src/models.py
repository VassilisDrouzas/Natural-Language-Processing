from abc import ABCMeta, abstractmethod
from collections import Counter
import itertools
from nltk.util import ngrams

START_TOKEN = "<start>"
END_TOKEN = "<end>"


class INgramModel:
    """
    An interface for all N-gram models.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, sentences_tokenized: list[list[str]]) -> None:
        """
        Train the model on a tokenized selection of sentences.
        :param sentences_tokenized: a list of all sentences. Each sentence is represented as a list of string tokens.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, tokenized_sentence: list[str]) -> str:
        """
        Predict the next word in a given sentence. Uses n-gram probability with Laplace Smoothing.
        :param tokenized_sentence: a list of string tokens
        :return: the most probable token
        """
        return ""


class BigramModel(INgramModel):
    """
    A basic bigram model using Laplace Smoothing.
    """

    def __init__(self, alpha: float):
        if alpha > 1.0 or alpha <= 0:
            raise ValueError(f"Alpha value must be between 0 and 1 exclusive (value given alpha={alpha})")

        self.vocab = {}
        self.alpha = alpha
        self.bigram_counter = Counter()
        self.unigram_counter = Counter()

    def fit(self, sentences_tokenized: list[list[str]]) -> None:
        self.vocab = set(itertools.chain.from_iterable(sentences_tokenized))

        for sentence in sentences_tokenized:
            self.unigram_counter.update(_process_ngrams(sentence, 1))
            self.bigram_counter.update(_process_ngrams(sentence, 2))

    def predict(self, tokenized_sentence: list[str]) -> str:

        if len(tokenized_sentence) == 0:
            raise ValueError("Cannot predict in empty sentence.")

        max_prob = -1
        max_token = None

        for token in self.vocab:
            prob = ((self.bigram_counter[(tokenized_sentence[-2], tokenized_sentence[-1])] + self.alpha) /
                    (self.unigram_counter[(token,)] + self.alpha * len(self.vocab)))

            if prob > max_prob:
                max_prob = prob
                max_token = token

        # we could also return the probability here?
        return max_token


def _process_ngrams(tokenized_sentence: list[str], ngram: int) -> list[tuple]:
    """
    Process a tokenized sentence into a list of ngrams.
    :param tokenized_sentence: a list of string tokens
    :param ngram: whether the ngrams will be unigrams, bigrams etc
    :return: a list of ngrams representing the original sentence
    """
    return [gram for gram in ngrams(tokenized_sentence, ngram, pad_left=True, pad_right=True,
                                    left_pad_symbol=START_TOKEN, right_pad_symbol=END_TOKEN)]
