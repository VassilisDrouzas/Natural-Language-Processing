import math
from typing import Callable, Any

import Levenshtein

from src.autocomplete import BigramModel, START_TOKEN


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
        assert len(original_tokenized_sentence) == len(target_tokenized_sentence), "Sentences must be of same length."

        lm_score = self.language_model.sentence_proba(target_tokenized_sentence)
        edit_score = sum([1 / (self.conditional_model(original_word, other_word) + 1)
                          for original_word, other_word in zip(original_tokenized_sentence, target_tokenized_sentence)])

        return self.lamda1 * math.log2(lm_score) + self.lamda2 * math.log2(edit_score)

    def generate_candidates(self, temp_sentence: list[str]) -> list[list[str]]:
        last_word = temp_sentence[-1]
        next_words = [word for (prev_word, word) in
                      self.language_model.bigram_counter.items() if prev_word == last_word]
        return [temp_sentence + [next_word] for next_word in next_words]

    def spell_correct(self, original_tokenized_sentence: list[str], max_depth: int, beam_width: int):
        def candidate_fn(state): return self.generate_candidates(state)
        def score_fn(candidate_sentence): return self.evaluate(original_tokenized_sentence, candidate_sentence)

        decoder = _SentenceBeamSearchDecoder(max_depth, beam_width, candidate_fn, score_fn)
        return decoder.search([START_TOKEN], 0.)


class _SentenceBeamSearchDecoder:
    """
    A Beam Search Decoder.
    """

    def __init__(self, max_depth: int, beam_width: int, candidate_generator_fn: Callable[[list[str]], list[list[str]]],
                 score_fn: Callable[[list[str]], float]):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.candidate_generator_fn = candidate_generator_fn
        self.score_fn = score_fn

    def search(self, initial_state: list[str], initial_state_score: float) -> Any:
        candidates = [(initial_state, initial_state_score)]

        for depth in range(self.max_depth):
            new_candidates = []
            for candidate, prob in candidates:
                for next_state in self.candidate_generator_fn(candidate):
                    new_prob = prob * self.score_fn(next_state)
                    new_candidates.append((next_state, new_prob))

            new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:self.beam_width]

        best_sequence, best_prob = max(candidates, key=lambda x: x[1])
        return best_sequence
