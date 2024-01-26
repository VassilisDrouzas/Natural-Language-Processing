from unittest import TestCase

from nltk import TweetTokenizer

from src.conditional_ngram_models import BigramSpellCorrector, TrigramSpellCorrector
from src.ngram_models import BigramModel, START_TOKEN, TrigramModel, END_TOKEN

# test bigram spell corrector
test_corpus = ["he plays football",
               "he plays football",
               "she enjoys good football",
               "she plays good music",
               "he prays to god",
               "please buy me the other ball,"
               "he pleases the other players by playing good football"]

tweet_wt = TweetTokenizer()
tokenized = [tweet_wt.tokenize(sentence) for sentence in test_corpus]


class TestBigramSpellCorrector(TestCase):

    def setUp(self):
        lang_model = BigramModel(alpha=0.01)
        lang_model.fit(tokenized)
        self.model = BigramSpellCorrector(lang_model, 0.5)

    def test_evaluate(self):
        text1 = self.model.language_model.format_input(tweet_wt.tokenize(test_corpus[3])) + [END_TOKEN]
        text2 = self.model.language_model.format_input(tweet_wt.tokenize(test_corpus[4])) + [END_TOKEN]

        eval1 = self.model.evaluate(text1, text1)
        eval2 = self.model.evaluate(text1, text2)

        self.assertLess(eval2, eval1)

    def test_generate_candidates(self):
        candidates = self.model.generate_candidates(tweet_wt.tokenize(START_TOKEN + " he plays"))
        self.assertGreater(len(candidates), 0)
        print(candidates)

    def test_spell_correct_unknown_sent(self):
        correction1 = self.model.spell_correct(tokenized[0], 2)
        self.assertIsNotNone(correction1)
        self.assertGreater(len(correction1), 0)
        print(f"Original: {tokenized[0]} \nModel: {correction1}")

        correction2 = self.model.spell_correct(tokenized[5], 2)
        self.assertIsNotNone(correction2)
        self.assertGreater(len(correction2), 0)
        print(f"Original: {tokenized[5]} \nModel: {correction2}")

        self.assertNotEqual(correction2, correction1)

    def test_spell_correct_typo(self):
        false_sent = "he prays god ftball"
        correction = self.model.spell_correct(tweet_wt.tokenize(false_sent), 2)
        self.assertIsNotNone(correction)
        self.assertGreater(len(correction), 0)
        print(f"Original: {false_sent} \nModel: {correction}")


class TestTrigramSpellCorrector(TestCase):

    def setUp(self):
        lang_model = TrigramModel(alpha=0.01)
        lang_model.fit(tokenized)
        self.model = TrigramSpellCorrector(lang_model, 0.5)

    def test_evaluate(self):
        text1 = self.model.language_model.format_input(tweet_wt.tokenize(test_corpus[3]))
        text2 = self.model.language_model.format_input(tweet_wt.tokenize(test_corpus[4]))

        eval1 = self.model.evaluate(text1, text1)
        eval2 = self.model.evaluate(text1, text2)

        self.assertLess(eval2, eval1)

    def test_generate_candidates(self):
        candidates = self.model.generate_candidates(tweet_wt.tokenize(START_TOKEN + " he plays"))
        self.assertGreater(len(candidates), 0)
        print(candidates)

    def test_spell_correct_unknown_sent(self):
        correction = self.model.spell_correct(tokenized[0], 2)
        self.assertIsNotNone(correction)
        self.assertGreater(len(correction), 0)
        print(f"Original: {tokenized[0]} \nModel: {correction}")

    def test_spell_correct_typo(self):
        false_sent = "he prays god ftball"
        correction = self.model.spell_correct(tweet_wt.tokenize(false_sent), 2)
        self.assertIsNotNone(correction)
        self.assertGreater(len(correction), 0)
        print(f"Original: {false_sent} \nModel: {correction}")

