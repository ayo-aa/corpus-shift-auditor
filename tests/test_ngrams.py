import math
import unittest

from corpus_shift_auditor.ngrams import NGramLanguageModel, tokenize


class NGramTests(unittest.TestCase):
    def test_tokenize_is_deterministic_and_keeps_punctuation(self):
        expected = ["ayo's", "model", "works", "—", "reliably", "!"]
        self.assertEqual(tokenize("Ayo's model works—reliably!"), expected)
        self.assertEqual(tokenize("Ayo's model works—reliably!"), expected)

    def test_probability_is_normalized_for_seen_context(self):
        model = NGramLanguageModel(order=2, alpha=0.2).fit(["a b a c"])
        probability_sum = sum(model.probability(("a", token)) for token in model.vocabulary)
        self.assertAlmostEqual(probability_sum, 1.0)

    def test_in_domain_text_has_lower_perplexity_than_unrelated_text(self):
        reference = ["refund order refund order refund order"]
        model = NGramLanguageModel(order=2, alpha=0.1).fit(reference)
        in_domain = model.perplexity(["refund order refund order"])
        shifted = model.perplexity(["gpu kernel checkpoint network"])
        self.assertTrue(math.isfinite(in_domain))
        self.assertLess(in_domain, shifted)

    def test_model_round_trip_preserves_perplexity(self):
        model = NGramLanguageModel(order=3, alpha=0.1).fit(
            ["one two three", "one two four"]
        )
        restored = NGramLanguageModel.from_dict(model.to_dict())
        self.assertAlmostEqual(
            restored.perplexity(["one two three"]),
            model.perplexity(["one two three"]),
        )

    def test_empty_corpus_is_rejected(self):
        model = NGramLanguageModel(order=2).fit(["some text"])
        with self.assertRaises(ValueError):
            model.perplexity([])


if __name__ == "__main__":
    unittest.main()

