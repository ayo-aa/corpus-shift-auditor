import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from corpus_shift_auditor.cli import main


class CliTests(unittest.TestCase):
    def test_cli_fit_and_audit_round_trip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.txt"
            incoming = root / "incoming.txt"
            model = root / "profile.json"
            report = root / "report.json"
            reference.write_text("refund order. refund package.", encoding="utf-8")
            incoming.write_text("gpu checkpoint network.", encoding="utf-8")

            self.assertEqual(
                main(["fit", str(reference), "--model", str(model), "--name", "demo"]),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        str(incoming),
                        "--model",
                        str(model),
                        "--output",
                        str(report),
                    ]
                ),
                0,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(payload["reference_name"], "demo")
        self.assertGreaterEqual(payload["risk_score"], 0)
        self.assertLessEqual(payload["risk_score"], 1)


if __name__ == "__main__":
    unittest.main()
