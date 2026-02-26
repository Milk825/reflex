import time
import unittest

from analysis.autonomy.integration import ConfidenceScorer
from analysis.autonomy.meta_cognition import MetaCognitionMonitor


class MetaCognitionHeuristicTests(unittest.TestCase):
    def test_repetition_loop_detected_by_third_failed_repeat_within_two_minutes(self):
        monitor = MetaCognitionMonitor()
        base = time.time()

        assessment = None
        for idx in range(3):
            assessment = monitor.update_after_action(
                {
                    "session": {"sessionId": "loop-1"},
                    "toolName": "exec",
                    "params": {"command": "cat ./missing-file.txt"},
                    "ok": False,
                    "confidence": 0.55 - idx * 0.05,
                },
                now=base + idx * 20,
            )

        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.state, "looping")
        self.assertGreaterEqual(assessment.triggers.get("repetition_loop", 0), 0.75)

    def test_backtracking_churn_detected_for_edit_revert_oscillation(self):
        monitor = MetaCognitionMonitor()
        base = time.time()

        actions = [
            {"oldText": "A", "newText": "B"},
            {"oldText": "B", "newText": "A"},
            {"oldText": "A", "newText": "B"},
            {"oldText": "B", "newText": "A"},
        ]

        assessment = None
        for idx, patch in enumerate(actions):
            assessment = monitor.update_after_action(
                {
                    "session": {"sessionId": "churn-1"},
                    "toolName": "edit",
                    "params": {"path": "notes.md", **patch},
                    "ok": False,
                    "confidence": 0.62 - idx * 0.06,
                },
                now=base + idx * 15,
            )

        self.assertIsNotNone(assessment)
        self.assertIn(assessment.state, {"confused", "uncertain"})
        self.assertGreaterEqual(assessment.triggers.get("backtracking_churn", 0), 0.45)

    def test_confidence_collapse_detection(self):
        monitor = MetaCognitionMonitor()
        base = time.time()

        confidences = [0.92, 0.84, 0.73, 0.61]
        assessment = None
        for idx, conf in enumerate(confidences):
            assessment = monitor.update_after_action(
                {
                    "session": {"sessionId": "collapse-1"},
                    "toolName": "read",
                    "params": {"path": "README.md"},
                    "ok": True,
                    "confidence": conf,
                },
                now=base + idx * 12,
            )

        self.assertIsNotNone(assessment)
        self.assertGreaterEqual(assessment.triggers.get("confidence_collapse", 0), 0.45)

    def test_over_eager_false_positive_rate_below_five_percent(self):
        monitor = MetaCognitionMonitor()
        base = time.time()

        flagged = 0
        scenarios = 120

        for scenario in range(scenarios):
            session = {"session": {"sessionId": f"fp-{scenario}"}}
            state = None
            for step in range(10):
                state = monitor.update_after_action(
                    {
                        **session,
                        "toolName": "exec",
                        "params": {"command": f"echo {scenario}-{step}"},
                        "ok": True,
                        "confidence": 0.72,
                        "estimated_actions": 4,
                        "progress": 0.42,
                        "batchOperation": True if step % 3 == 0 else False,
                        "longWorkflowRequested": True if step % 4 == 0 else False,
                    },
                    now=base + scenario * 5 + step,
                )

            if state and state.state == "over_eager":
                flagged += 1

        false_positive_rate = flagged / scenarios
        self.assertLess(false_positive_rate, 0.05)

    def test_health_update_can_affect_meta_state(self):
        monitor = MetaCognitionMonitor()
        assessment = monitor.on_health_update(
            {
                "session": {"sessionId": "health-1"},
                "health": {"normalized": 0.0},
            }
        )

        self.assertIn(assessment.state, {"uncertain", "confused"})

    def test_meta_update_mean_under_two_ms(self):
        monitor = MetaCognitionMonitor()
        iterations = 3500

        started = time.perf_counter()
        now = time.time()
        for idx in range(iterations):
            monitor.update_after_action(
                {
                    "session": {"sessionId": "perf-1"},
                    "toolName": "read" if idx % 2 == 0 else "exec",
                    "params": {"path": f"file-{idx % 10}.md"},
                    "ok": idx % 5 != 0,
                    "confidence": 0.8 - (idx % 7) * 0.05,
                    "progress": 0.3 if idx % 5 else 0.1,
                    "estimated_actions": 8,
                },
                now=now + idx * 0.2,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        mean_ms = elapsed_ms / iterations
        self.assertLess(mean_ms, 2.0)


class MetaPenaltyScoringTests(unittest.TestCase):
    def test_meta_penalty_has_ten_percent_weight_in_score_formula(self):
        scorer = ConfidenceScorer(default_temperature=1.0)

        low_penalty = scorer.score_before_action(
            p_hist=0.75,
            health=0.8,
            risk=0.2,
            ambiguity=0.2,
            complexity=0.2,
            meta_penalty=0.0,
            sample_size=30,
            temperature=1.0,
        )
        high_penalty = scorer.score_before_action(
            p_hist=0.75,
            health=0.8,
            risk=0.2,
            ambiguity=0.2,
            complexity=0.2,
            meta_penalty=0.8,
            sample_size=30,
            temperature=1.0,
        )

        # 10% weight means an 0.8 penalty should reduce raw score by ~0.08 (before calibration/sample effects).
        self.assertAlmostEqual(low_penalty.raw_score - high_penalty.raw_score, 0.08, places=2)
        self.assertGreater(low_penalty.confidence, high_penalty.confidence)


if __name__ == "__main__":
    unittest.main()
