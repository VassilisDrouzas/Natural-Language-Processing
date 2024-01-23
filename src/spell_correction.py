import math
from abc import abstractmethod
from typing import Callable

import Levenshtein

from src.autocomplete import BigramModel, START_TOKEN, BaseNgramModel, TrigramModel
from src.beam_search import SentenceBeamSearchDecoder


class BaseSpellCorrector:
    """
    Abstract base class for all n-gram-based spell checking models.
    These models take into account a given sentence and predict the best correction.
    """

    def __init__(self, language_model: BaseNgramModel, lamda1: float, lamda2: float,
                 conditional_model: Callable[[str, str], float] = Levenshtein.distance):
        """
        Initialize the BaseSpellCorrector.
        :param language_model: The n-gram language model used for prediction.
        :param lamda1: Weight for language model score in the final evaluation.
        :param lamda2: Weight for edit distance score in the final evaluation.
        :param conditional_model: The conditional model used for edit distance calculation.
                                  Defaults to Levenshtein distance.
        """
        self.language_model = language_model
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.conditional_model = conditional_model

    @abstractmethod
    def _initial_search_state(self) -> list[str]:
        """
        Abstract method to define the initial search state for the spell correction.
        :return: The initial search state as a list of strings.
        """
        pass

    @abstractmethod
    def generate_candidates(self, temp_sentence: list[str]) -> list[list[str]]:
        """
        Abstract method to generate candidate corrections given a temporary sentence.
        :param temp_sentence: The temporary sentence as a list of strings.
        :return: List of candidate corrections, each represented as a list of strings.
        """
        pass

    def spell_correct(self, original_tokenized_sentence: list[str], max_depth: int, beam_width: int) -> list[str]:
        """
        Perform spell correction using beam search.
        :param original_tokenized_sentence: The original sentence as a list of strings.
        :param max_depth: Maximum depth for beam search.
        :param beam_width: Width of the beam for beam search.
        :return: The corrected sentence as a list of strings.
        """
        def candidate_fn(state): return self.generate_candidates(state)

        def score_fn(candidate_sentence): return self.evaluate(
            self.language_model.format_input(original_tokenized_sentence),
            candidate_sentence)

        decoder = SentenceBeamSearchDecoder(max_depth, beam_width, candidate_fn, score_fn)
        return decoder.search(self._initial_search_state(), 0.)

    def evaluate(self, original_tokenized_sentence: list[str], target_tokenized_sentence: list[str]) -> float:
        """
        Evaluate the candidate correction based on language model score and edit distance.
        :param original_tokenized_sentence: The original sentence as a list of strings.
        :param target_tokenized_sentence: The candidate correction as a list of strings.
        :return: The log-probability of the sentence given the target sentence.
        """
        clipped_length = min(len(original_tokenized_sentence), len(target_tokenized_sentence))
        clipped_original_sentence = original_tokenized_sentence[:clipped_length]
        clipped_target_sentence = target_tokenized_sentence[:clipped_length]

        lm_score = self.language_model.sentence_proba(clipped_target_sentence)
        edit_score = sum([-math.log2(self.conditional_model(original_word, other_word) + 1)
                          for original_word, other_word in zip(clipped_original_sentence, clipped_target_sentence)])

        return self.lamda1 * lm_score + self.lamda2 * edit_score


class BigramSpellCorrector(BaseSpellCorrector):
    """
    Spell corrector based on a Bigram language model.
    """

    def __init__(self, language_model: BigramModel, lamda1: float, lamda2: float,
                 conditional_model: Callable[[str, str], float] = Levenshtein.distance):
        """
        Initialize the BigramSpellCorrector.
        :param language_model: The Bigram language model used for prediction.
        :param lamda1: Weight for language model score in the final evaluation.
        :param lamda2: Weight for edit distance score in the final evaluation.
        :param conditional_model: The conditional model used for edit distance calculation.
                                  Defaults to Levenshtein distance.
        :raises ValueError: If the provided language_model is not an instance of BigramModel.
        """
        super().__init__(language_model, lamda1, lamda2, conditional_model)

        #if not isinstance(language_model, BigramModel):                                                            #had to comment out otherwise it would raise the error on my notebook
            #raise ValueError("The Bigram spell corrector needs a bigram model to function properly.")

    def generate_candidates(self, temp_sentence: list[str]) -> list[list[str]]:
        """
        Generate candidate corrections based on the last word in the temporary sentence.
        :param temp_sentence: The temporary sentence as a list of strings.
        :return: List of candidate corrections, each represented as a list of strings.
        """
        last_word = temp_sentence[-1]
        next_words = [word for ((prev_word, word), occ) in
                      self.language_model.bigram_counter.items() if prev_word == last_word]
        return [temp_sentence + [next_word] for next_word in next_words]

    def _initial_search_state(self) -> list[str]:
        """
        Define the initial search state for BigramSpellCorrector.
        :return: The initial search state as a list of strings.
        """
        return [START_TOKEN]


class TrigramSpellCorrector(BaseSpellCorrector):
    """
    Spell corrector based on Trigram language model.
    """

    def __init__(self, language_model: BigramModel, lamda1: float, lamda2: float,
                 conditional_model: Callable[[str, str], float] = Levenshtein.distance):
        """
        Initialize the TrigramSpellCorrector.
        :param language_model: The Trigram language model used for prediction.
        :param lamda1: Weight for language model score in the final evaluation.
        :param lamda2: Weight for edit distance score in the final evaluation.
        :param conditional_model: The conditional model used for edit distance calculation.
                                  Defaults to Levenshtein distance.
        :raises ValueError: If the provided language_model is not an instance of TrigramModel.
        """
        super().__init__(language_model, lamda1, lamda2, conditional_model)

        if not isinstance(language_model, TrigramModel):
            raise ValueError("The Trigram spell corrector needs a trigram model to function properly.")

    def generate_candidates(self, temp_sentence: list[str]) -> list[list[str]]:
        """
        Generate candidate corrections based on the last two words in the temporary sentence.
        :param temp_sentence: The temporary sentence as a list of strings.
        :return: List of candidate corrections, each represented as a list of strings.
        """
        last_word = temp_sentence[-1]
        second_last_word = temp_sentence[-2]
        next_words = [word3 for ((word1, word2, word3), occ) in
                      self.language_model.trigram_counter.items() if word1 == second_last_word and word2 == last_word]
        return [temp_sentence + [next_word] for next_word in next_words]

    def _initial_search_state(self) -> list[str]:
        """
        Define the initial search state for TrigramSpellCorrector.
        :return: The initial search state as a list of strings.
        """
        return [START_TOKEN, START_TOKEN]