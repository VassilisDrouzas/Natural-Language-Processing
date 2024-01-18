from unittest import TestCase
from src.models import BigramModel, TrigramModel
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
        assert model.vocab_len == 21

    def test_predict(self):
        model = BigramModel(alpha=0.01)
        model.fit(tokenized)

        self.assertRaises(ValueError, model.predict, [])
        self.assertRaises(AssertionError, model.predict, None)

        # for now just make sure it produces something
        self.assertIsNotNone(model.predict(tweet_wt.tokenize("he plays")))

        # prediction = model.predict(tweet_wt.tokenize("he plays"))
        # self.assertEqual("football", prediction)


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
        assert len(model.vocab) == 21

    def test_predict(self):
        model = TrigramModel(alpha=0.01)
        model.fit(tokenized)

        self.assertRaises(ValueError, model.predict, [])
        self.assertRaises(ValueError, model.predict, ["he"])
        self.assertRaises(AssertionError, model.predict, None)

        # for now just make sure it produces something
        self.assertIsNotNone(model.predict(tweet_wt.tokenize("he plays")))

        # prediction = model.predict(tweet_wt.tokenize("he plays"))
        # self.assertEqual("football", prediction)
