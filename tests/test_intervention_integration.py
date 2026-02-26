import unittest

from analysis.autonomy.core.integration import AutonomyCoordinator


class InterventionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_circuit_breaker_blocks_or_switches_failing_tool(self):
        coordinator = AutonomyCoordinator(mode="active")

        for _ in range(3):
            await coordinator._process_after_action(
                {
                    "session": {"sessionId": "circuit-breaker"},
                    "toolName": "exec",
                    "params": {"command": "cat missing.txt"},
                    "ok": False,
                    "errorType": "ENOENT",
                }
            )

        decision = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "circuit-breaker"},
                "toolName": "exec",
                "params": {"command": "cat missing.txt"},
                "predictive": {"p_hist": 0.95, "sample_size": 80},
                "health": {"normalized": 0.9},
                "ambiguity": 0.1,
                "complexity": 0.15,
            }
        )

        self.assertEqual(decision.metadata["intervention"]["type"], "tool")
        self.assertEqual(decision.metadata["intervention"]["priority"], "CRITICAL")
        self.assertIn(decision.directive, {"BLOCK", "ASK_CONFIRMATION"})

    async def test_fallback_tool_suggested_when_primary_is_unstable(self):
        coordinator = AutonomyCoordinator(mode="active")

        for _ in range(2):
            await coordinator._process_after_action(
                {
                    "session": {"sessionId": "fallback"},
                    "toolName": "exec",
                    "params": {"command": "cat missing.txt"},
                    "ok": False,
                }
            )

        decision = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "fallback"},
                "toolName": "exec",
                "params": {"command": "cat missing.txt"},
                "predictive": {"p_hist": 0.88, "sample_size": 30},
                "health": {"normalized": 0.85},
                "ambiguity": 0.2,
                "complexity": 0.25,
            }
        )

        self.assertEqual(decision.directive, "SWITCH_ALTERNATIVE")
        self.assertEqual(decision.metadata.get("alternativeTool"), "read")
        self.assertEqual(decision.metadata["intervention"]["type"], "tool")

    async def test_config_change_triggers_verification(self):
        coordinator = AutonomyCoordinator(mode="active")

        decision = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "config-verify"},
                "toolName": "write",
                "params": {"path": "config.json", "content": "{}"},
                "predictive": {"p_hist": 0.96, "sample_size": 100},
                "health": {"normalized": 0.95},
                "ambiguity": 0.1,
                "complexity": 0.2,
            }
        )

        self.assertEqual(decision.directive, "ASK_CONFIRMATION")
        self.assertEqual(decision.metadata["intervention"]["type"], "config")

    async def test_priority_selection_prefers_high_over_medium(self):
        coordinator = AutonomyCoordinator(mode="active")

        decision = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "priority"},
                "toolName": "write",
                "params": {"path": "config.json", "content": "{}"},
                "predictive": {"p_hist": 0.58, "sample_size": 10},
                "health": {"normalized": 0.7},
                "ambiguity": 0.15,
                "complexity": 0.2,
            }
        )

        self.assertEqual(decision.metadata["intervention"]["type"], "config")
        self.assertEqual(decision.metadata["intervention"]["priority"], "HIGH")

    async def test_health_degradation_triggers_adaptation(self):
        coordinator = AutonomyCoordinator(mode="active")

        await coordinator.on_health_update(
            {
                "session": {"sessionId": "health-adapt"},
                "health": {"normalized": 0.28},
            }
        )

        decision = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "health-adapt"},
                "toolName": "read",
                "params": {"path": "README.md"},
                "predictive": {"p_hist": 0.95, "sample_size": 120},
                "ambiguity": 0.1,
                "complexity": 0.1,
            }
        )

        self.assertEqual(decision.metadata["adaptive"]["profile"], "conservative")
        self.assertEqual(decision.metadata["intervention"]["type"], "interaction")
        self.assertEqual(decision.directive, "ASK_CONFIRMATION")

    async def test_intervention_lifecycle_resolves_after_outcome(self):
        coordinator = AutonomyCoordinator(mode="active")

        decision = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "lifecycle"},
                "toolName": "write",
                "params": {"path": "config.json", "content": "{}"},
                "predictive": {"p_hist": 0.9, "sample_size": 20},
                "health": {"normalized": 0.9},
                "ambiguity": 0.2,
                "complexity": 0.2,
            }
        )

        intervention = decision.metadata["intervention"]
        self.assertEqual(intervention["status"], "active")

        await coordinator._process_after_action(
            {
                "session": {"sessionId": "lifecycle"},
                "toolName": "write",
                "params": {"path": "config.json", "content": "{}"},
                "ok": True,
                "interventionId": intervention["id"],
            }
        )

        active = coordinator.current_intervention_state({"session": {"sessionId": "lifecycle"}})
        self.assertEqual(active["count"], 0)

    async def test_intervention_override_beats_confidence_directive(self):
        coordinator = AutonomyCoordinator(mode="active")

        # Build a likely PROCEED baseline first.
        baseline = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "override"},
                "toolName": "read",
                "params": {"path": "README.md"},
                "predictive": {"p_hist": 0.98, "sample_size": 150},
                "health": {"normalized": 0.95},
                "ambiguity": 0.05,
                "complexity": 0.05,
            }
        )
        self.assertIn(baseline.directive, {"PROCEED", "PROCEED_WITH_CAUTION"})

        # Degrade health to force critical interaction intervention override.
        await coordinator.on_health_update(
            {
                "session": {"sessionId": "override"},
                "health": {"normalized": 0.2},
            }
        )

        overridden = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "override"},
                "toolName": "read",
                "params": {"path": "README.md"},
                "predictive": {"p_hist": 0.98, "sample_size": 150},
                "ambiguity": 0.05,
                "complexity": 0.05,
            }
        )

        self.assertEqual(overridden.metadata["intervention"]["priority"], "CRITICAL")
        self.assertEqual(overridden.directive, "ASK_CONFIRMATION")

    async def test_decision_intervention_type_present(self):
        coordinator = AutonomyCoordinator(mode="active")

        decision = await coordinator.before_action(
            {
                "mode": "active",
                "session": {"sessionId": "decision-intervention"},
                "toolName": "exec",
                "params": {"command": "shutdown --help"},
                "predictive": {"p_hist": 0.42, "sample_size": 20},
                "health": {"normalized": 0.65},
                "ambiguity": 0.2,
                "complexity": 0.2,
            }
        )

        self.assertEqual(decision.metadata["intervention"]["type"], "decision")
        self.assertEqual(decision.directive, "ASK_CONFIRMATION")


if __name__ == "__main__":
    unittest.main()
