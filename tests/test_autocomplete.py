from unittest import TestCase
from src.autocomplete import (BigramModel, TrigramModel, LinearInterpolationModel, _process_ngrams, START_TOKEN,
                              END_TOKEN)
from nltk.tokenize import TweetTokenizer

test_corpus = ["he plays football",
               "he plays football",
               "she enjoys good football",
               "she plays good music",
               "he prays to god",
               "please buy me the other ball,"
               "he pleases the other players by playing good football"]

tweet_wt = TweetTokenizer()
tokenized = [tweet_wt.tokenize(sentence) for sentence in test_corpus]


class TestBigramModel(TestCase):

    def test_init(self):
        self.assertRaises(ValueError, BigramModel, 34)
        self.assertRaises(ValueError, BigramModel, 0)
        self.assertRaises(ValueError, BigramModel, -2)

        BigramModel(alpha=1)
        model = BigramModel(alpha=0.01)
        assert model.alpha == 0.01

    def test_fit(self):
        model = BigramModel(alpha=0.01)

        model.fit([])

        model.fit(tokenized)
        # vocab + meta-tokens
        assert len(model.vocabulary()) == 21 + 2

    def test_predict(self):
        model = BigramModel(alpha=0.01)

        self.assertRaises(RuntimeError, model.predict, ["he plays"])

        model.fit(tokenized)

        self.assertRaises(AssertionError, model.predict, None)

        # for now just make sure it produces something
        model.predict([])
        self.assertIsNotNone(model.predict(tweet_wt.tokenize("he plays")))

        # prediction = model.predict(tweet_wt.tokenize("he plays"))
        # self.assertEqual("football", prediction)

    def test_prediction_proba(self):
        model = BigramModel(alpha=0.01)
        self.assertRaises(RuntimeError, model.prediction_proba, tweet_wt.tokenize(test_corpus[0][:-3]), "football")

        model.fit(tokenized)
        probs = model.prediction_proba(tweet_wt.tokenize(test_corpus[0][:-3]), "football")
        assert probs <= 0

    def test_sentence_proba(self):
        model = BigramModel(alpha=0.01)
        self.assertRaises(AssertionError, model.sentence_proba, None)
        self.assertRaises(RuntimeError, model.sentence_proba, test_corpus[0])

        model.fit(tokenized)
        correct_sentence_probs = model.sentence_proba(tweet_wt.tokenize(test_corpus[0]))
        self.assertLess(correct_sentence_probs, 0)

        false_sentence_probs = model.sentence_proba(tweet_wt.tokenize("she pleases te ball"))
        self.assertLess(false_sentence_probs, 0)

        self.assertLess(false_sentence_probs, correct_sentence_probs)


class TestTrigramModel(TestCase):

    def test_init(self):
        self.assertRaises(ValueError, TrigramModel, 34)
        self.assertRaises(ValueError, TrigramModel, 0)
        self.assertRaises(ValueError, TrigramModel, -2)

        TrigramModel(alpha=1)
        model = TrigramModel(alpha=0.01)
        assert model.alpha == 0.01

    def test_fit(self):
        model = TrigramModel(alpha=0.01)

        model.fit([])

        model.fit(tokenized)
        self.assertEqual(21 + 2, len(model.vocabulary()))

    def test_predict(self):
        model = TrigramModel(alpha=0.01)

        self.assertRaises(RuntimeError, model.predict, tweet_wt.tokenize("he plays"))

        model.fit(tokenized)
        self.assertRaises(AssertionError, model.predict, None)

        # for now just make sure it produces something
        model.predict([])
        model.predict(["he"])
        self.assertIsNotNone(model.predict(tweet_wt.tokenize("he plays")))

        # prediction = model.predict(tweet_wt.tokenize("he plays"))
        # self.assertEqual("football", prediction)

    def test_prediction_proba(self):
        model = TrigramModel(alpha=0.01)
        self.assertRaises(RuntimeError, model.prediction_proba, tweet_wt.tokenize(test_corpus[0][:-3]), "football")

        model.fit(tokenized)
        probs = model.prediction_proba(tweet_wt.tokenize(test_corpus[0][:-3]), "football")
        assert probs <= 0

    def test_sentence_proba(self):
        model = TrigramModel(alpha=0.01)
        self.assertRaises(AssertionError, model.sentence_proba, None)
        self.assertRaises(RuntimeError, model.sentence_proba, test_corpus[0])

        model.fit(tokenized)
        correct_sentence_probs = model.sentence_proba(tweet_wt.tokenize(test_corpus[0]))

        false_sentence_probs = model.sentence_proba(tweet_wt.tokenize("she pleases te ball"))

        assert correct_sentence_probs > false_sentence_probs


class TestLinearInterpolationModel(TestCase):

    def test_init(self):
        self.assertRaises(ValueError, LinearInterpolationModel, 34, 0.5)
        self.assertRaises(ValueError, LinearInterpolationModel, 0, 0.5)
        self.assertRaises(ValueError, LinearInterpolationModel, -2, 0.5)

        self.assertRaises(ValueError, LinearInterpolationModel, 0.1, 2)
        self.assertRaises(ValueError, LinearInterpolationModel, 0.1, -1)
        self.assertRaises(ValueError, LinearInterpolationModel, 0.1, 345)

        LinearInterpolationModel(alpha=1, lamda=1)
        LinearInterpolationModel(alpha=0.01, lamda=0.5)

    def test_fit(self):
        model = LinearInterpolationModel(alpha=0.01, lamda=0.5)
        model.fit([])
        model.fit(tokenized)
        self.assertEqual(len(model.vocabulary()), 21 + 2)

    def test_predict(self):
        model = LinearInterpolationModel(alpha=0.01, lamda=0.5)

        self.assertRaises(RuntimeError, model.predict, ["he plays"])

        model.fit(tokenized)
        self.assertRaises(AssertionError, model.predict, None)

        # for now just make sure it produces something
        model.predict([])
        model.predict(["he"])
        self.assertIsNotNone(model.predict(tweet_wt.tokenize("he plays")))

    def test_prediction_proba(self):
        model = LinearInterpolationModel(alpha=0.01, lamda=0.5)
        self.assertRaises(RuntimeError, model.prediction_proba, tweet_wt.tokenize(test_corpus[0][:-3]), "football")

        model.fit(tokenized)
        probs = model.prediction_proba(tweet_wt.tokenize(test_corpus[0][:-3]), "football")
        assert probs <= 0

    def test_sentence_proba(self):
        model = LinearInterpolationModel(alpha=0.01, lamda=0.5)
        self.assertRaises(AssertionError, model.sentence_proba, None)
        self.assertRaises(RuntimeError, model.sentence_proba, test_corpus[0])

        model.fit(tokenized)
        correct_sentence_probs = model.sentence_proba(tweet_wt.tokenize(test_corpus[0]))

        false_sentence_probs = model.sentence_proba(tweet_wt.tokenize("she pleases te ball"))

        assert correct_sentence_probs > false_sentence_probs


class Test(TestCase):
    def test_1grams(self):
        processed = _process_ngrams(tokenized[0], 1)
        assert processed[0][0] == START_TOKEN
        assert processed[-1][-1] == END_TOKEN

    def test_process_2grams(self):
        processed = _process_ngrams(tokenized[0], 2)
        assert processed[0][0] == START_TOKEN
        assert processed[-1][-1] == END_TOKEN

    def test_process_3grams(self):
        processed = _process_ngrams(tokenized[0], 3)
        assert processed[0][0] == START_TOKEN
        assert processed[1][0] == START_TOKEN
        assert processed[-1][-1] == END_TOKEN
