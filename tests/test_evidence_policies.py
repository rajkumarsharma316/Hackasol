import unittest

import numpy as np

from vtrace import confidence, fusion, matcher, metrics, reid
from vtrace.ocr import _prep, _project_alphanumeric_posteriors
from vtrace.pipeline import _predict_box
from tools.evaluate_benchmark import decision_correct, measure


class EvidencePolicyTests(unittest.TestCase):
    def test_plate_metrics_use_reference_cer_and_longer_length_similarity(self):
        scored = metrics.plate_metrics([("A", "AB"), ("AB", "A")])
        self.assertEqual(scored["character_errors"], 2)
        self.assertAlmostEqual(scored["character_error_rate"], 2 / 3)
        self.assertAlmostEqual(scored["normalized_edit_similarity"], 0.5)
        self.assertAlmostEqual(scored["macro_character_error_rate"], 0.75)

    def test_ocr_preparation_matches_easyocr_right_padding_width(self):
        prepared = _prep(np.full((64, 167), 128, dtype=np.uint8))
        self.assertEqual(tuple(prepared.shape), (1, 1, 64, 192))

    def test_unsupported_ocr_symbols_become_blank_instead_of_false_characters(self):
        class Reader:
            character = "a!"

        prob = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
        projected = _project_alphanumeric_posteriors(prob, Reader())
        self.assertEqual(int(projected.argmax(1)[0]), 0)
        self.assertAlmostEqual(float(projected[0].sum()), 1.0)
        self.assertAlmostEqual(float(projected[0, 11]), 0.2)  # A is CHARS index 10.

    def test_plate_metrics_normalize_format_but_preserve_unknowns_and_empty_reads(self):
        scored = metrics.plate_metrics([(" mh 12-de-4521 ", "MH12DE4521"), ("", "AB12")])
        self.assertEqual(scored["exact_matches"], 1)
        self.assertEqual(scored["character_errors"], 4)
        self.assertEqual(scored["reference_characters"], 14)

    def test_missing_truth_is_excluded_and_reported(self):
        scored = metrics.plate_metrics([("A", None), ("", "AB")])
        self.assertEqual(scored["samples"], 1)
        self.assertEqual(scored["missing_ground_truth"], 1)
        self.assertEqual(scored["character_error_rate"], 1.0)
        self.assertIsNone(decision_correct({"system_decision": "no_confident_match"}, {}))
        self.assertTrue(decision_correct({"system_decision": "no_confident_match"},
                                         {"in_registry": False}))

    def test_unlabeled_registry_status_is_not_counted_as_correct(self):
        result = {"match": {"status": "no_confident_match"}}
        self.assertIsNone(decision_correct(result, {}))
        self.assertEqual(measure([{"clip": "x", "plate": "", "system_decision": "no_confident_match"}],
                                 {"x": {}})["decision_labels_available"], 0)

    def test_decision_accuracy_handles_raw_results_and_summarized_rows(self):
        truth = {"a": {"plate": "MH12DE4521", "in_registry": True},
                 "b": {"plate": "UP32ZZ9087", "in_registry": False}}
        rows = [
            {"clip": "a", "system_decision": "match", "top_candidate": "mh 12 de 4521"},
            {"clip": "b", "system_decision": "no_confident_match", "registry_decision": "no_confident_match"},
        ]
        self.assertEqual(measure(rows, truth)["decision_accuracy"], "2/2")

    def test_empty_registry_abstains(self):
        result = matcher.match(matcher.Evidence(), [])
        self.assertEqual(result["status"], "no_confident_match")
        self.assertEqual(result["candidates"], [])

    def test_fusion_does_not_align_glyphs_across_two_line_plate_rows(self):
        row0 = {"dist": np.eye(36, dtype=np.float32)[10], "char": "A", "line": 0}
        row1 = {"dist": np.eye(36, dtype=np.float32)[10], "char": "A", "line": 1}
        self.assertEqual(fusion._sim(row0, row1), -1.5)
        self.assertGreater(fusion._sim(row0, {**row0, "char": "B"}), -1.5)

    def test_tracker_extrapolates_recent_motion(self):
        track = {"boxes": [(0, [0, 0, 10, 10], 0.8), (1, [2, 0, 12, 10], 0.9)]}
        self.assertEqual(_predict_box(track, 2), [4.0, 0.0, 14.0, 10.0])

    def test_weak_match_is_retained_as_review_candidate(self):
        match = {"status": "match", "candidates": [{"confidence": 85}]}
        quality = {"tier": "medium", "automatic_eligible": False}
        gated = confidence.gate_registry_match(match, quality)
        self.assertEqual(gated["status"], "review_required")
        self.assertEqual(gated["registry_status"], "match")

    def test_high_tier_requires_multiple_independent_signals(self):
        positions = [{"known": True, "conf": 0.82} for _ in range(7)]
        frames = [{"plate_source": "plate model", "loc_score": 0.8,
                   "used_in_fusion": True, "weight": 0.2, "mean_conf": 0.8}
                  for _ in range(4)]
        score = confidence.assess(
            {"_boxes": [(i, [0, 0, 20, 20], 0.9) for i in range(4)]},
            frames, {"positions": positions, "n_frames": 4},
            {"status": "match", "candidates": [{"confidence": 70}]},
        )
        self.assertEqual(score["tier"], "high")
        self.assertTrue(score["automatic_eligible"])
        self.assertIn("not statistically calibrated", score["method"])

    def test_cross_camera_appearance_alone_never_confirms_identity(self):
        observation = {
            "fusion": {"positions": [], "text": "", "n_frames": 0},
            "attributes": {"colour": {"white": 1.0}, "body_type": {"car": 1.0}, "make_model": {}},
            "signature": {"embedding": np.ones(8).tolist()},
            "video": {"file": "same-looking-camera.mp4"},
        }
        result = reid.compare_clips(observation, observation, [])
        self.assertNotEqual(result["verdict"], "same_vehicle")
        self.assertEqual(result["verdict"], "possible_same_vehicle")


if __name__ == "__main__":
    unittest.main()
