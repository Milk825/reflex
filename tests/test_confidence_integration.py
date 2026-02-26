import math
import time
import unittest

from analysis.autonomy.core.integration import AutonomyCoordinator, ConfidenceScorer
from analysis.autonomy.core.preflight import PreflightChecker


class ConfidenceScorerTests(unittest.TestCase):
    def test_formula_and_signal_wiring(self):
        scorer = ConfidenceScorer(default_temperature=1.0)

        result = scorer.score_before_action(
            p_hist=0.8,
            health=0.9,
            risk=0.2,
            ambiguity=0.1,
            complexity=0.2,
            meta_penalty=0.0,
            sample_size=20,
            temperature=1.0,
        )

        expected_unpenalized = (
            0.35 * 0.8
            + 0.20 * 0.9
            + 0.15 * (1.0 - 0.2)
            + 0.10 * (1.0 - 0.1)
            + 0.10 * (1.0 - 0.2)
            + 0.10 * (1.0 - 0.0)
        )
        expected_penalty = min(0.15, 0.15 * math.exp(-20 / 5.0))
        expected = max(0.01, min(0.99, expected_unpenalized - expected_penalty))

        self.assertAlmostEqual(result.raw_score, expected, places=6)
        self.assertAlmostEqual(result.confidence, expected, places=6)
        self.assertEqual(result.band, "high")

    def test_sample_size_penalty_reduces_confidence(self):
        scorer = ConfidenceScorer(default_temperature=1.0)

        sparse = scorer.score_before_action(
            p_hist=0.75,
            health=0.8,
            risk=0.25,
            ambiguity=0.2,
            complexity=0.2,
            meta_penalty=0.1,
            sample_size=0,
        )
        dense = scorer.score_before_action(
            p_hist=0.75,
            health=0.8,
            risk=0.25,
            ambiguity=0.2,
            complexity=0.2,
            meta_penalty=0.1,
            sample_size=25,
        )

        self.assertLess(sparse.confidence, dense.confidence)
        self.assertGreater(dense.confidence - sparse.confidence, 0.10)

    def test_temperature_calibration_softens_extreme_scores(self):
        scorer = ConfidenceScorer(default_temperature=1.0)

        base = scorer.score_before_action(
            p_hist=0.85,
            health=0.9,
            risk=0.1,
            ambiguity=0.1,
            complexity=0.15,
            meta_penalty=0.0,
            sample_size=15,
            temperature=1.0,
        )
        hotter = scorer.score_before_action(
            p_hist=0.85,
            health=0.9,
            risk=0.1,
            ambiguity=0.1,
            complexity=0.15,
            meta_penalty=0.0,
            sample_size=15,
            temperature=1.8,
        )

        self.assertLess(hotter.confidence, base.confidence)

    def test_confidence_scoring_mean_under_2ms(self):
        scorer = ConfidenceScorer(default_temperature=1.0)

        iterations = 4000
        started = time.perf_counter()
        for _ in range(iterations):
            scorer.score_before_action(
                p_hist=0.72,
                health=0.81,
                risk=0.38,
                ambiguity=0.21,
                complexity=0.34,
                meta_penalty=0.08,
                sample_size=9,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        mean_ms = elapsed_ms / iterations

        self.assertLess(mean_ms, 2.0)


class PreflightTests(unittest.TestCase):
    def test_preflight_detects_protected_file(self):
        checker = PreflightChecker()
        result = checker.check_before_action("write", {"path": "~/workspace/MEMORY.md", "content": "x"})

        self.assertTrue(result.has_protected_file)
        self.assertGreaterEqual(result.risk, 0.70)
        self.assertIn("high", [w.severity for w in result.warnings])

    def test_preflight_detects_destructive_pattern_and_critical_combo(self):
        checker = PreflightChecker()
        result = checker.check_before_action("exec", {"command": "rm -rf ~/.openclaw/workspace/MEMORY.md"})

        self.assertTrue(result.has_destructive_pattern)
        self.assertTrue(result.has_protected_file)
        self.assertTrue(result.critical)
        self.assertGreaterEqual(result.risk, 0.95)


class DirectiveRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_critical_preflight_blocks(self):
        coordinator = AutonomyCoordinator(mode="active")
        decision = await coordinator.before_action(
            {
                "toolName": "exec",
                "params": {"command": "rm -rf ~/.openclaw/workspace/MEMORY.md"},
                "mode": "active",
                "predictive": {"p_hist": 0.7, "sample_size": 25},
                "health": {"overall_score": 90},
            }
        )

        self.assertEqual(decision.directive, "BLOCK")

    async def test_medium_band_with_risk_routes_to_caution(self):
        coordinator = AutonomyCoordinator(mode="active")
        decision = await coordinator.before_action(
            {
                "toolName": "exec",
                "params": {"command": "shutdown --help"},
                "mode": "active",
                "predictive": {"p_hist": 0.78, "sample_size": 50},
                "health": {"overall_score": 82},
                "ambiguity": 0.2,
                "complexity": 0.2,
            }
        )

        self.assertEqual(decision.directive, "PROCEED_WITH_CAUTION")
        self.assertGreaterEqual(decision.metadata["risk"], 0.55)

    async def test_low_confidence_routes_to_clarification_or_switch(self):
        coordinator = AutonomyCoordinator(mode="active")
        decision = await coordinator.before_action(
            {
                "toolName": "exec",
                "params": {"command": "cat ./notes.txt"},
                "mode": "active",
                "predictive": {"p_hist": 0.35, "sample_size": 1},
                "health": {"overall_score": 60},
                "ambiguity": 0.45,
                "complexity": 0.55,
                "meta_penalty": 0.35,
            }
        )

        self.assertIn(decision.directive, {"SWITCH_ALTERNATIVE", "ASK_CLARIFICATION"})

    async def test_very_low_confidence_asks_confirmation(self):
        coordinator = AutonomyCoordinator(mode="active")
        decision = await coordinator.before_action(
            {
                "toolName": "browser",
                "params": {"targetUrl": "https://example.com"},
                "mode": "active",
                "predictive": {"p_hist": 0.2, "sample_size": 0},
                "health": {"overall_score": 45},
                "ambiguity": 0.9,
                "complexity": 0.9,
                "meta_penalty": 0.9,
            }
        )

        self.assertEqual(decision.directive, "ASK_CONFIRMATION")

    async def test_protected_file_non_destructive_asks_confirmation(self):
        coordinator = AutonomyCoordinator(mode="active")
        decision = await coordinator.before_action(
            {
                "toolName": "write",
                "params": {"path": "SOUL.md", "content": "update"},
                "mode": "active",
                "predictive": {"p_hist": 0.92, "sample_size": 60},
                "health": {"overall_score": 95},
                "ambiguity": 0.1,
                "complexity": 0.2,
            }
        )

        self.assertEqual(decision.directive, "ASK_CONFIRMATION")

    async def test_looping_meta_state_escalates_to_clarification(self):
        coordinator = AutonomyCoordinator(mode="active")

        now = time.time()
        for idx in range(3):
            coordinator._meta_monitor.update_after_action(
                {
                    "session": {"sessionId": "routing-loop"},
                    "toolName": "exec",
                    "params": {"command": "cat ./missing.txt"},
                    "ok": False,
                    "confidence": 0.6 - idx * 0.05,
                },
                now=now + idx * 15,
            )

        decision = await coordinator.before_action(
            {
                "toolName": "exec",
                "params": {"command": "cat ./missing.txt"},
                "mode": "active",
                "session": {"sessionId": "routing-loop"},
                "predictive": {"p_hist": 0.8, "sample_size": 20},
                "health": {"overall_score": 90},
                "ambiguity": 0.2,
                "complexity": 0.4,
            }
        )

        self.assertEqual(decision.directive, "ASK_CLARIFICATION")
        self.assertEqual(decision.metadata["meta"]["state"], "looping")

    async def test_meta_penalty_from_monitor_is_applied_to_confidence(self):
        coordinator = AutonomyCoordinator(mode="active")
        now = time.time()

        for idx in range(9):
            coordinator._meta_monitor.update_after_action(
                {
                    "session": {"sessionId": "over-eager"},
                    "toolName": "exec",
                    "params": {"command": f"step-{idx}"},
                    "ok": False,
                    "confidence": 0.7,
                    "estimated_actions": 3,
                    "progress": 0.1,
                },
                now=now + idx,
            )

        penalized = await coordinator.before_action(
            {
                "toolName": "exec",
                "params": {"command": "final-step"},
                "mode": "active",
                "session": {"sessionId": "over-eager"},
                "predictive": {"p_hist": 0.86, "sample_size": 40},
                "health": {"overall_score": 90},
                "ambiguity": 0.2,
                "complexity": 0.3,
            }
        )

        baseline = await coordinator.before_action(
            {
                "toolName": "exec",
                "params": {"command": "final-step"},
                "mode": "active",
                "session": {"sessionId": "baseline"},
                "predictive": {"p_hist": 0.86, "sample_size": 40},
                "health": {"overall_score": 90},
                "ambiguity": 0.2,
                "complexity": 0.3,
            }
        )

        self.assertGreater(penalized.metadata["signals"]["meta_penalty"], baseline.metadata["signals"]["meta_penalty"])
        self.assertLess(penalized.confidence, baseline.confidence)

    async def test_coordinator_mode_is_authoritative_without_override_flag(self):
        coordinator = AutonomyCoordinator(mode="suggest")

        decision = await coordinator.before_action(
            {
                "toolName": "read",
                "params": {"path": "README.md"},
                "mode": "active",
                "predictive": {"p_hist": 0.9, "sample_size": 10},
                "health": {"overall_score": 90},
                "ambiguity": 0.1,
                "complexity": 0.1,
            }
        )

        self.assertEqual(decision.metadata["mode"], "suggest")

        overridden = await coordinator.before_action(
            {
                "toolName": "read",
                "params": {"path": "README.md"},
                "mode": "active",
                "allowModeOverride": True,
                "predictive": {"p_hist": 0.9, "sample_size": 10},
                "health": {"overall_score": 90},
                "ambiguity": 0.1,
                "complexity": 0.1,
            }
        )

        self.assertEqual(overridden.metadata["mode"], "active")


if __name__ == "__main__":
    unittest.main()
