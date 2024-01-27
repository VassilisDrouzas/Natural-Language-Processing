import math
from abc import abstractmethod
from typing import Callable

import Levenshtein

from src.ngram_models import BigramModel, START_TOKEN, BaseNgramModel, TrigramModel, END_TOKEN, UNKNOWN_TOKEN
from src.beam_search import SentenceBeamSearchDecoder


class BaseSpellCorrector:
    """
    Abstract base class for all n-gram-based spell checking models.
    These models take into account a given sentence and predict the best correction.
    """

    def __init__(self, language_model: BaseNgramModel, lamda: float,
                 conditional_model: Callable[[str, str], float] = Levenshtein.distance):
        """
        Initialize the BaseSpellCorrector.
        :param language_model: The n-gram language model used for prediction.
        :param lamda: Weight for language model score in the final evaluation.
        The weight for the edit distance score will correspondingly be 1-lambda. Values must be between 0 and 1.
        :param conditional_model: The conditional model used for edit distance calculation.
                                  Defaults to Levenshtein distance.
        """
        if lamda > 1.0 or lamda <= 0:
            raise ValueError(f"Lamda value must be between 0 (exclusive) and 1 (value given alpha={lamda})")

        self.language_model = language_model
        self.lamda1 = lamda
        self.lamda2 = 1 - lamda
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

    def spell_correct(self, original_tokenized_sentence: list[str], beam_width: int) -> list[str]:
        """
        Perform spell correction using beam search.
        :param original_tokenized_sentence: The original sentence as a list of strings.
        :param beam_width: Width of the beam for beam search.
        :return: The corrected sentence as a list of strings.
        """
        def candidate_fn(state): return self.generate_candidates(state)

        def score_fn(candidate_sentence): return self.evaluate(formatted_sentence, candidate_sentence)

        words_to_be_guessed = len(original_tokenized_sentence) + 1
        formatted_sentence = self.language_model.format_input(original_tokenized_sentence) + [END_TOKEN]
        decoder = SentenceBeamSearchDecoder(words_to_be_guessed, beam_width, candidate_fn, score_fn)
        response = decoder.search(self._initial_search_state(), 0.)
        # remove meta tokens
        return [token for token in response if token != START_TOKEN and token != END_TOKEN]

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

        # if unknown token, disregard edit distance on this token
        no_unk_original = []
        no_unk_target = []
        for i in range(len(clipped_original_sentence)):
            curr_original_token = clipped_original_sentence[i]
            curr_target_token = clipped_target_sentence[i]
            if curr_original_token != UNKNOWN_TOKEN:
                no_unk_original.append(curr_original_token)
                no_unk_target.append(curr_target_token)

        edit_score = -sum([math.log2(self.conditional_model(original_word, other_word) + 1)
                          for original_word, other_word in zip(no_unk_original, no_unk_target)])


        return self.lamda1 * lm_score + self.lamda2 * edit_score


class BigramSpellCorrector(BaseSpellCorrector):
    """
    Spell corrector based on a Bigram language model.
    """

    def __init__(self, language_model: BigramModel, lamda: float,
                 conditional_model: Callable[[str, str], float] = Levenshtein.distance):
        """
        Initialize the BigramSpellCorrector.
        :param language_model: The n-gram language model used for prediction.
        :param lamda: Weight for language model score in the final evaluation.
        The weight for the edit distance score will correspondingly be 1-lambda. Values must be between 0 and 1.
        :param conditional_model: The conditional model used for edit distance calculation.
                                  Defaults to Levenshtein distance.
        """
        super().__init__(language_model, lamda, conditional_model)

        if not isinstance(language_model, BigramModel):                                                            #had to comment out otherwise it would raise the error on my notebook
            raise ValueError("The Bigram spell corrector needs a bigram model to function properly.")

    def generate_candidates(self, temp_sentence: list[str]) -> list[list[str]]:
        """
        Generate candidate corrections based on the last word in the temporary sentence.
        :param temp_sentence: The temporary sentence as a list of strings.
        :return: List of candidate corrections, each represented as a list of strings.
        """
        last_word = temp_sentence[-1]
        next_words = [word for ((prev_word, word), occ) in
                      self.language_model.bigram_counter.items() if prev_word == last_word and word != UNKNOWN_TOKEN]
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

    def __init__(self, language_model: TrigramModel, lamda: float,
                 conditional_model: Callable[[str, str], float] = Levenshtein.distance):
        """
        Initialize the TrigramSpellCorrector.
        :param language_model: The n-gram language model used for prediction.
        :param lamda: Weight for language model score in the final evaluation.
        The weight for the edit distance score will correspondingly be 1-lambda. Values must be between 0 and 1.
        :param conditional_model: The conditional model used for edit distance calculation.
                                  Defaults to Levenshtein distance.
        """
        super().__init__(language_model, lamda, conditional_model)

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
        next_words = [word3 for ((word1, word2, word3), occ) in self.language_model.trigram_counter.items()
                      if word1 == second_last_word and word2 == last_word and word3 != UNKNOWN_TOKEN]
        return [temp_sentence + [next_word] for next_word in next_words]

    def _initial_search_state(self) -> list[str]:
        """
        Define the initial search state for TrigramSpellCorrector.
        :return: The initial search state as a list of strings.
        """
        return [START_TOKEN, START_TOKEN]
