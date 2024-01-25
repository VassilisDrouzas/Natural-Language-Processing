from typing import Callable


class SentenceBeamSearchDecoder:
    """
    A Beam Search Decoder. Used internally for the various language models.
    """

    def __init__(self, max_depth: int, beam_width: int, candidate_generator_fn: Callable[[list[str]], list[list[str]]],
                 score_fn: Callable[[list[str]], float]):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.candidate_generator_fn = candidate_generator_fn
        self.score_fn = score_fn

    def search(self, initial_state: list[str], initial_state_score: float) -> list[str]:
        candidates = [(initial_state, initial_state_score)]

        for depth in range(self.max_depth):
            new_candidates = []
            for candidate, prob in candidates:
                for next_state in self.candidate_generator_fn(candidate):
                    new_prob = prob + self.score_fn(next_state)
                    new_candidates.append((next_state, new_prob))

            new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:self.beam_width]

        if len(candidates) == 0:
            raise ValueError("Can not build sentence: No suitable candidates found.")

        best_sequence, best_prob = max(candidates, key=lambda x: x[1])
        return best_sequence
