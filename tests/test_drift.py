import unittest

from corpus_shift_auditor import audit_corpus, build_profile
from corpus_shift_auditor.drift import CorpusProfile


REFERENCE = [
    "Customers requested refunds for delayed orders.",
    "Support agents reviewed orders and confirmed shipping status.",
    "The team sent replacement products for lost packages.",
]


class DriftTests(unittest.TestCase):
    def test_profile_can_be_saved_and_loaded(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        profile = build_profile(REFERENCE, name="support")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            restored = CorpusProfile.load(path)
        self.assertEqual(restored.name, "support")
        self.assertEqual(restored.token_counts, profile.token_counts)
        self.assertAlmostEqual(restored.reference_perplexity, profile.reference_perplexity)

    def test_identical_corpus_has_no_lexical_divergence(self):
        profile = build_profile(REFERENCE)
        report = audit_corpus(profile, REFERENCE)
        self.assertAlmostEqual(report.lexical_js_divergence, 0.0)
        self.assertAlmostEqual(report.out_of_vocabulary_rate, 0.0)
        self.assertEqual(report.risk_level, "low")

    def test_unrelated_corpus_scores_higher_than_similar_corpus(self):
        profile = build_profile(REFERENCE)
        similar = audit_corpus(
            profile, ["An agent reviewed a delayed order and issued a refund."]
        )
        shifted = audit_corpus(
            profile,
            ["Distributed GPU workers synchronized gradients across accelerator nodes."],
        )
        self.assertGreater(shifted.risk_score, similar.risk_score)
        self.assertGreater(shifted.out_of_vocabulary_rate, similar.out_of_vocabulary_rate)

    def test_report_is_json_serializable_shape(self):
        profile = build_profile(REFERENCE, name="support")
        payload = audit_corpus(profile, ["A replacement order shipped."]).to_dict()
        self.assertEqual(payload["reference_name"], "support")
        self.assertIn(payload["risk_level"], {"low", "moderate", "high", "critical"})
        self.assertIn("incoming_structure", payload)


if __name__ == "__main__":
    unittest.main()

