# test example by Foivos Anagnostou
from unittest import TestCase

from src.ngram_models import START_TOKEN
from src.beam_search import SentenceBeamSearchDecoder


# test example adapted from Foivos Charalampakos
bigram_model = {
    (START_TOKEN, 'I'): 0.5,
    (START_TOKEN, 'He'): 0.3,
    (START_TOKEN, 'She'): 0.3,
    (START_TOKEN, 'It'): 0.35,
    ('He', 'am'): 0.0005,
    ('He', 'is'): 0.4,
    ('She', 'is'): 0.4,
    ('It', 'is'): 0.4,
    ('I', 'am'): 0.5,
    ('I', 'have'): 0.48,
    ('am', 'a'): 0.7,
    ('have', 'a'): 0.8,
    ('is', 'a'): 0.8,
    ('am', 'not'): 0.3,
    ('a', 'cat'): 0.6,
    ('a', 'dog'): 0.4,
    ('have', 'a'): 0.8,
    ('have', 'an'): 0.2,
    ('an', 'apple'): 0.9,
    ('an', 'orange'): 0.1,
    ('cat', 'and'): 0.4,
    ('and', 'a'): 0.45,
    ('cat', 'is'): 0.4,
    ('dog', 'and'): 0.5,
    ('dog', 'barks'): 0.5,
    ('barks', 'loudly'): 0.7,
    ('barks', 'quietly'): 0.3,
    ('I', 'like'): 0.4,
    ('I', 'dislike'): 0.6,
    ('like', 'to'): 0.8,
    ('like', 'the'): 0.2,
    ('the', 'sun'): 0.5,
    ('the', 'moon'): 0.5,
    ('moon', 'is'): 0.9,
    ('moon', 'bright'): 0.1,
    ('sun', 'shines'): 0.7,
    ('sun', 'sets'): 0.3,
}


def generate_candidates(state):
    # Given state , generate possible next words
    last_word = state[-1]
    next_words = [word for (prev_word, word) in bigram_model if prev_word == last_word]
    return [state + [next_word] for next_word in next_words]


def score(state):
    # Calculate the probability of the word sequence using the bigram model
    probability = 1.0
    for i in range(1, len(state)):
        prev_word, word = state[i - 1], state[i]
        probability *= bigram_model.get((prev_word, word), 0.0)
    return probability


class TestSentenceBeamSearchDecoder(TestCase):
    def test_search(self):
        decoder = SentenceBeamSearchDecoder(max_depth=10, beam_width=2,
                                             candidate_generator_fn=generate_candidates, score_fn=score)
        result = decoder.search([START_TOKEN], 0.0)
        print(result)
        self.assertIsNotNone(result)
        self.assertNotEqual([START_TOKEN], result)

