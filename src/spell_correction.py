import math
from typing import Callable

import Levenshtein

from src.autocomplete import BigramModel, START_TOKEN
from src.beam_search import SentenceBeamSearchDecoder


class BigramSpellCorrector:

    def __init__(self, language_model: BigramModel, lamda1: float, lamda2: float,
                 conditional_model: Callable[[str, str], float] = Levenshtein.distance):
        self.language_model = language_model
        self.lamda1 = lamda1
        self.lamda2 = lamda2

        # this could perhaps be improved if we used normalized distance
        # https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance
        self.conditional_model = conditional_model

    def evaluate(self, original_tokenized_sentence: list[str], target_tokenized_sentence: list[str]) -> float:
        # clip sentences to same length
        clipped_length = min(len(original_tokenized_sentence), len(target_tokenized_sentence))
        clipped_original_sentence = original_tokenized_sentence[:clipped_length]
        clipped_target_sentence = target_tokenized_sentence[:clipped_length]

        # compute probability scores
        lm_score = self.language_model.sentence_proba(clipped_target_sentence)
        edit_score = sum([-math.log2(self.conditional_model(original_word, other_word) + 1)
                          for original_word, other_word in zip(clipped_original_sentence, clipped_target_sentence)])

        return self.lamda1 * lm_score + self.lamda2 * edit_score

    def generate_candidates(self, temp_sentence: list[str]) -> list[list[str]]:
        last_word = temp_sentence[-1]
        next_words = [word for ((prev_word, word), occ) in
                      self.language_model.bigram_counter.items() if prev_word == last_word]
        return [temp_sentence + [next_word] for next_word in next_words]

    def spell_correct(self, original_tokenized_sentence: list[str], max_depth: int, beam_width: int):
        def candidate_fn(state): return self.generate_candidates(state)

        def score_fn(candidate_sentence): return self.evaluate(
            self.language_model.format_input(original_tokenized_sentence),
            candidate_sentence)

        decoder = SentenceBeamSearchDecoder(max_depth, beam_width, candidate_fn, score_fn)
        return decoder.search([START_TOKEN], 0.)
