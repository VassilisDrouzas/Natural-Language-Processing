import itertools
import more_itertools
import math
from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Collection

from nltk.util import ngrams

# different start symbols aren't supported in nltk's ngrams method
START_TOKEN = "<start>"
END_TOKEN = "<end>"


class BaseNgramModel:
    """
    Base class for all n-gram models.
    """

    def __init__(self, alpha: float):
        if alpha > 1.0 or alpha <= 0:
            raise ValueError(f"Alpha value must be between 0 (exclusive) and 1 (value given alpha={alpha})")
        self.alpha = alpha

        self.trained = False

    @abstractmethod
    def fit(self, sentences_tokenized: list[list[str]]) -> None:
        """
        Train the model on a tokenized selection of sentences.
        :param sentences_tokenized: a list of all sentences. Each sentence is represented as a list of string tokens.
        :return: None
        """
        self.trained = True

    @abstractmethod
    def vocabulary(self) -> Collection[str]:
        """
        Get all tokens from the model's vocabulary.
        :return: a list of all tokens in the model
        """
        pass

    def predict(self, tokenized_sentence: list[str]) -> str:
        """
        Predict the next word in a given sentence. Uses n-gram probability with Laplace Smoothing.
        :param tokenized_sentence: a list of string tokens
        :raise RuntimeError: if the model has not been trained
        :return: the most probable token
        """
        assert tokenized_sentence is not None

        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        # since we are only looking for the next word, we need not compute the probabilities of all ngrams,
        # since only the last one changes
        formatted_sentence = self.format_input(tokenized_sentence)
        max_prob = - math.inf
        max_token = None

        for token in self.vocabulary():
            prob = self.prediction_proba(formatted_sentence, token)

            if prob > max_prob:
                max_prob = prob
                max_token = token

        return max_token

    @abstractmethod
    def prediction_proba(self, tokenized_sentence: list[str], token: str) -> float:
        """
        Get the model's log-probability for a specific token given a sentence.
        :param tokenized_sentence: a list of string tokens
        :param token: the token
        :raise RuntimeError: if the model has not been trained
        :return: the log2-probability that the token is next
        """
        assert tokenized_sentence is not None

        if not self.trained:
            raise RuntimeError("Model has not been trained.")
        return 0

    @abstractmethod
    def sentence_proba(self, tokenized_sentence: list[str]) -> float:
        """
        Calculate the log-probability of an entire sentence.
        :param tokenized_sentence: a list of string tokens
        :raise RuntimeError: if the model has not been trained
        :return: the log2-probability of the sentence
        """
        assert tokenized_sentence is not None
        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        return 0

    @abstractmethod
    def format_input(self, tokenized_sentence: list[str]) -> list[str]:
        """
        Format the sentence as to include proper START and END tags.
        :param tokenized_sentence: a list of string tokens
        :return: the sentence with the appropriate START and END tags
        """
        return []


class BigramModel(BaseNgramModel):
    """
    A basic bigram model using Laplace Smoothing.
    """

    def __init__(self, alpha: float):
        """
        Create a bigram model.
        :param alpha: the Laplace smoothing parameter. Must be between 0 and 1 (excluding 0)
        """
        super().__init__(alpha)

        self.bigram_counter = Counter()
        self.unigram_counter = Counter()

    def fit(self, sentences_tokenized: list[list[str]]) -> None:
        super().fit(sentences_tokenized)

        for sentence in sentences_tokenized:
            # get the strings inside the tuples
            self.unigram_counter.update([tuple_[0] for tuple_ in _process_ngrams(sentence, 1)])
            self.bigram_counter.update(_process_ngrams(sentence, 2))

    def prediction_proba(self, tokenized_sentence: list[str], token: str) -> float:
        super().prediction_proba(tokenized_sentence, token)

        return (math.log2((self.bigram_counter[(tokenized_sentence[-1], token)] + self.alpha)) -
                math.log2(self.unigram_counter[token] + self.alpha * len(self.vocabulary())))

    def sentence_proba(self, tokenized_sentence: list[str]) -> float:
        super().sentence_proba(tokenized_sentence)
        formatted_sentence = self.format_input(tokenized_sentence)

        probs = 0
        for token1, token2 in more_itertools.windowed(formatted_sentence, 2):
            probs += self.prediction_proba([token1], token2)
        return probs

    def format_input(self, tokenized_sentence: list[str]) -> list[str]:
        return [START_TOKEN] + tokenized_sentence + [END_TOKEN]

    def vocabulary(self) -> Collection[str]:
        return self.unigram_counter.keys()


class TrigramModel(BaseNgramModel):
    """
    A basic trigram model using Laplace Smoothing.
    """

    def __init__(self, alpha: float):
        """
        Create a trigram model.
        :param alpha: the Laplace smoothing parameter. Must be between 0 and 1 (excluding 0)
        """
        super().__init__(alpha)

        self.vocab = {}
        self.bigram_counter = Counter()
        self.trigram_counter = Counter()

    def fit(self, sentences_tokenized: list[list[str]]) -> None:
        super().fit(sentences_tokenized)

        self.vocab = set(itertools.chain.from_iterable(sentences_tokenized)).union({START_TOKEN, END_TOKEN})

        for sentence in sentences_tokenized:
            self.bigram_counter.update(_process_ngrams(sentence, 2))
            self.trigram_counter.update(_process_ngrams(sentence, 3))

    def prediction_proba(self, tokenized_sentence: list[str], token: str) -> float:
        super().prediction_proba(tokenized_sentence, token)

        return (math.log2(self.trigram_counter[(tokenized_sentence[-2], tokenized_sentence[-1], token)] + self.alpha) -
                math.log2(self.bigram_counter[(tokenized_sentence[-1], token)] + self.alpha * len(self.vocab)))

    def sentence_proba(self, tokenized_sentence: list[str]) -> float:
        super().sentence_proba(tokenized_sentence)
        formatted_sentence = self.format_input(tokenized_sentence)

        probs = 0
        for token1, token2, token3 in more_itertools.windowed(formatted_sentence, 3):
            probs += self.prediction_proba([token1, token2], token3)
        return probs

    def format_input(self, tokenized_sentence: list[str]) -> list[str]:
        return [START_TOKEN] + [START_TOKEN] + tokenized_sentence + [END_TOKEN]

    def vocabulary(self) -> Collection[str]:
        return self.vocab


# I could generalize this to support combinations of unigrams, bigrams and trigrams, but we'll see
class LinearInterpolationModel(BaseNgramModel):
    """
    A model using linear interpolation between a bigram and trigram model.
    """

    def __init__(self, alpha: float, lamda: float):
        """
        Create a linear interpolation model between a bigram and trigram model.
        :param alpha: the Laplace smoothing parameter for the internal models. Must be between 0 and 1 (excluding 0)
        :param lamda: the interpolation parameter, where probability = lambda * (bigram probability)
        + (1-lamda) * (trigram probability)
        """
        super().__init__(alpha)

        if lamda > 1.0 or lamda <= 0:
            raise ValueError(f"Lamda value must be between 0 (exclusive) and 1 (value given lamda={lamda})")
        self.lamda = lamda

        self.bigram_model = BigramModel(alpha)
        self.trigram_model = TrigramModel(alpha)

    def fit(self, sentences_tokenized: list[list[str]]) -> None:
        super().fit(sentences_tokenized)
        self.bigram_model.fit(sentences_tokenized)
        self.trigram_model.fit(sentences_tokenized)

    def prediction_proba(self, tokenized_sentence: list[str], token: str) -> float:
        """
        Get the interpolation's weighted probability sum for a specific token given a sentence.
        :param tokenized_sentence: a list of string tokens
        :param token: the token
        :raise Runtime Error: if the model has not been trained
        :return: the weighted probability that the token is next
        """
        super().prediction_proba(tokenized_sentence, token)

        bigram_format = self.bigram_model.format_input(tokenized_sentence)
        bigram_prob = self.bigram_model.prediction_proba(bigram_format, token)

        trigram_format = self.trigram_model.format_input(tokenized_sentence)
        trigram_prob = self.trigram_model.prediction_proba(trigram_format, token)
        return self.lamda * bigram_prob + (1 - self.lamda) * trigram_prob

    def sentence_proba(self, tokenized_sentence: list[str]) -> float:
        super().sentence_proba(tokenized_sentence)

        return (self.lamda * self.bigram_model.sentence_proba(tokenized_sentence)
                + (1 - self.lamda) * self.trigram_model.sentence_proba(tokenized_sentence))

    def vocabulary(self) -> Collection[str]:
        return set(self.bigram_model.vocabulary()).union(set(self.trigram_model.vocabulary()))

    def format_input(self, tokenized_sentence: list[str]) -> list[str]:
        return tokenized_sentence


def _process_ngrams(tokenized_sentence: list[str], ngram: int) -> list[tuple]:
    """
    Process a tokenized sentence into a list of ngrams.
    :param tokenized_sentence: a list of string tokens
    :param ngram: whether the ngrams will be unigrams, bigrams etc
    :return: a list of ngrams representing the original sentence
    """

    ngram_sent = [gram for gram in ngrams(tokenized_sentence, ngram, pad_left=True, pad_right=True,
                                          left_pad_symbol=START_TOKEN, right_pad_symbol=END_TOKEN)]
    if ngram == 1:
        return [tuple([START_TOKEN], )] + ngram_sent + [tuple([END_TOKEN], )]
    else:
        return ngram_sent
