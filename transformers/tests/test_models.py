from unittest import TestCase

from ..tasks.models import BaselineLabelClassifier, _record_label


EXAMPLE_TEXT = "the quick brown fox jumps over the lazy dog".split()
EXAMPLE_LABELS = ["DT", "JJ", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN"]


class TestBaselineLabelClassifier(TestCase):
    def test_fit(self):
        cls = BaselineLabelClassifier().fit(EXAMPLE_TEXT, EXAMPLE_LABELS)
        self.assertEqual(cls.most_popular_pos_tag, "JJ")
        self.assertNotEqual(cls.word_pos_dict, {})

    def test_predict(self):
        cls = BaselineLabelClassifier()
        self.assertRaises(Exception, cls.predict(["the"]))

        cls = cls.fit(EXAMPLE_TEXT, EXAMPLE_LABELS)
        self.assertEqual(cls.predict("unknown".split()), ["JJ"])
        self.assertEqual(cls.predict("the quick".split()), ["DT", "JJ"])


class Test(TestCase):
    def test__record_label(self):
        dict_ = {}
        _record_label(dict_, EXAMPLE_TEXT[0], EXAMPLE_LABELS[0])
        self.assertEqual(dict_[EXAMPLE_TEXT[0]][EXAMPLE_LABELS[0]], 1)

        _record_label(dict_, EXAMPLE_TEXT[0], EXAMPLE_LABELS[0])
        self.assertEqual(dict_[EXAMPLE_TEXT[0]][EXAMPLE_LABELS[0]], 2)



