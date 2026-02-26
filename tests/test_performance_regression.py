import statistics
import time
import unittest

from analysis.autonomy.integration import AutonomyCoordinator


class PerformanceRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_before_action_p95_under_10ms_for_10k_actions(self):
        coordinator = AutonomyCoordinator()

        latencies = []
        context = {
            "toolName": "read",
            "params": {"path": "README.md"},
            "predictive": {"p_hist": 0.83, "sample_size": 20},
            "health": {"normalized": 0.9},
            "ambiguity": 0.12,
            "complexity": 0.1,
            "mode": "shadow",
        }

        for _ in range(10_000):
            started = time.perf_counter()
            await coordinator.before_action(context)
            latencies.append((time.perf_counter() - started) * 1000.0)

        latencies.sort()
        p95_index = int(len(latencies) * 0.95) - 1
        p95 = latencies[max(0, p95_index)]
        mean = statistics.fmean(latencies)

        self.assertLess(p95, 10.0, msg=f"p95={p95:.3f}ms mean={mean:.3f}ms")

    async def test_noncritical_prompt_rate_capped_at_one_third(self):
        coordinator = AutonomyCoordinator(mode="suggest")

        prompts = 0
        turns = 60
        for _ in range(turns):
            decision = await coordinator.before_action(
                {
                    "toolName": "exec",
                    "params": {"command": "cat unknown.txt"},
                    "predictive": {"p_hist": 0.35, "sample_size": 1},
                    "health": {"normalized": 0.8},
                    "ambiguity": 0.5,
                    "complexity": 0.4,
                    "mode": "suggest",
                }
            )
            comm = decision.metadata.get("communication") if isinstance(decision.metadata.get("communication"), dict) else {}
            if bool(comm.get("noncriticalPrompt")) and bool(comm.get("shouldMessage")):
                prompts += 1

        self.assertLessEqual(prompts / turns, 1.0 / 3.0)

    async def test_high_risk_low_confidence_always_requires_confirmation(self):
        coordinator = AutonomyCoordinator()

        confirmations = 0
        total = 40
        for _ in range(total):
            decision = await coordinator.before_action(
                {
                    "toolName": "exec",
                    "params": {"command": "shutdown now"},
                    "predictive": {"p_hist": 0.45, "sample_size": 1},
                    "health": {"normalized": 0.7},
                    "ambiguity": 0.7,
                    "complexity": 0.7,
                }
            )
            if decision.directive == "ASK_CONFIRMATION":
                confirmations += 1

        self.assertEqual(confirmations, total)


if __name__ == "__main__":
    unittest.main()
