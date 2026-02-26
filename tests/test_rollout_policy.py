import unittest

from analysis.autonomy.rollout import RolloutTracker


class RolloutPolicyTests(unittest.TestCase):
    def test_shadow_to_suggest_gate_passes_with_required_metrics(self):
        tracker = RolloutTracker()

        for _ in range(1000):
            tracker.record_preflight_overhead(4.2)

        # Fail-open always proceeds.
        for _ in range(10):
            tracker.record_fail_open(proceeded=True)

        # Over-eager FP under 5%.
        for i in range(100):
            tracker.record_decision(
                mode="shadow",
                override=False,
                proactive_prompt=False,
                noncritical_prompt=False,
                over_eager_alert=True,
                over_eager_false_positive=i < 4,
            )

        gate = tracker.evaluate_transition(current_mode="shadow", target_mode="suggest")
        self.assertTrue(gate.allowed)

    def test_shadow_to_suggest_gate_blocks_on_bad_metrics(self):
        tracker = RolloutTracker()

        for _ in range(100):
            tracker.record_preflight_overhead(12.5)

        tracker.record_fail_open(proceeded=False)

        for i in range(20):
            tracker.record_decision(
                mode="shadow",
                override=False,
                proactive_prompt=False,
                noncritical_prompt=False,
                over_eager_alert=True,
                over_eager_false_positive=i < 5,
            )

        gate = tracker.evaluate_transition(current_mode="shadow", target_mode="suggest")
        self.assertFalse(gate.allowed)
        self.assertGreaterEqual(len(gate.reasons), 1)

    def test_suggest_to_active_gate_checks_prompt_override_and_brier(self):
        tracker = RolloutTracker()

        # Calibrate shadow baseline (good brier).
        for _ in range(100):
            tracker.record_calibration(mode="shadow", predicted_confidence=0.8, success=True)

        # Suggest has low prompt rate and override rate.
        for idx in range(120):
            tracker.record_decision(
                mode="suggest",
                override=idx < 20,  # 16.7%
                proactive_prompt=idx % 5 == 0,
                noncritical_prompt=idx % 5 == 0,  # 20%
                over_eager_alert=False,
                over_eager_false_positive=False,
            )

        # Suggest brier <= shadow brier.
        for _ in range(100):
            tracker.record_calibration(mode="suggest", predicted_confidence=0.82, success=True)

        gate = tracker.evaluate_transition(current_mode="suggest", target_mode="active")
        self.assertTrue(gate.allowed)

    def test_suggest_to_active_gate_blocks_if_prompt_rate_too_high(self):
        tracker = RolloutTracker()

        for _ in range(40):
            tracker.record_calibration(mode="shadow", predicted_confidence=0.8, success=True)
            tracker.record_calibration(mode="suggest", predicted_confidence=0.8, success=True)

        for idx in range(30):
            tracker.record_decision(
                mode="suggest",
                override=False,
                proactive_prompt=True,
                noncritical_prompt=idx < 15,  # 50% > 1/3
                over_eager_alert=False,
                over_eager_false_positive=False,
            )

        gate = tracker.evaluate_transition(current_mode="suggest", target_mode="active")
        self.assertFalse(gate.allowed)
        self.assertTrue(any("prompt_rate" in reason for reason in gate.reasons))


if __name__ == "__main__":
    unittest.main()
