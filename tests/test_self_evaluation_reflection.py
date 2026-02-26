import asyncio
import time
import unittest

from analysis.autonomy.core.auto_reflection import AutoReflectionEngine
from analysis.autonomy.core.integration import AutonomyCoordinator
from analysis.autonomy.core.self_evaluation import SelfEvaluator


class SelfEvaluatorTests(unittest.TestCase):
    def test_micro_eval_mean_under_two_ms(self):
        evaluator = SelfEvaluator()

        iterations = 3500
        started = time.perf_counter()
        for _ in range(iterations):
            evaluator.micro_eval(
                {
                    "toolName": "read",
                    "ok": True,
                    "durationMs": 8,
                    "expectedDurationMs": 12,
                    "confidence": 0.78,
                }
            )

        mean_ms = ((time.perf_counter() - started) * 1000.0) / iterations
        self.assertLess(mean_ms, 2.0)
        evaluator.stop()

    def test_should_trigger_deep_eval_for_significant_events(self):
        evaluator = SelfEvaluator()

        result = evaluator.micro_eval(
            {
                "session": {"sessionId": "deep-1"},
                "toolName": "exec",
                "ok": False,
                "durationMs": 250,
                "expectedDurationMs": 30,
                "retries": 2,
                "confidence": 0.92,
            }
        )

        self.assertTrue(result.deep_eval_queued)
        self.assertIn("failure", result.deep_eval_reasons)
        self.assertIn("retries>=2", result.deep_eval_reasons)
        self.assertIn("confidence_outcome_mismatch", result.deep_eval_reasons)
        evaluator.stop()

    def test_deep_eval_enqueue_non_blocking(self):
        evaluator = SelfEvaluator()

        for _ in range(120):
            evaluator.micro_eval(
                {
                    "session": {"sessionId": "queue-1"},
                    "toolName": "exec",
                    "ok": False,
                    "durationMs": 90,
                    "retries": 2,
                    "confidence": 0.9,
                }
            )

        stats = evaluator.stats()
        self.assertEqual(stats["deep_eval_sync_wait_over_1ms"], 0)
        evaluator.stop()


class ReflectionGuardrailTests(unittest.TestCase):
    def test_guardrails_one_per_task_and_spacing(self):
        engine = AutoReflectionEngine(min_spacing_seconds=300)
        now = time.time()
        context = {
            "session": {"sessionId": "r-1"},
            "taskBoundary": True,
            "taskId": "task-1",
            "sessionSummary": {
                "taskId": "task-1",
                "actionCount": 5,
                "successCount": 4,
                "failureCount": 1,
                "timeSpentMs": 180000,
            },
        }

        first = engine.evaluate(context, now=now)
        self.assertIsNotNone(first)

        duplicate = engine.evaluate(context, now=now + 10)
        self.assertIsNone(duplicate)

        too_soon_other_task = engine.evaluate(
            {
                **context,
                "taskId": "task-2",
                "sessionSummary": {**context["sessionSummary"], "taskId": "task-2"},
            },
            now=now + 60,
        )
        self.assertIsNone(too_soon_other_task)

        allowed_later = engine.evaluate(
            {
                **context,
                "taskId": "task-3",
                "sessionSummary": {**context["sessionSummary"], "taskId": "task-3"},
            },
            now=now + 301,
        )
        self.assertIsNotNone(allowed_later)


class EndToEndReflectionLessonTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_boundary_to_reflection_to_lesson(self):
        coordinator = AutonomyCoordinator()

        session = {"sessionId": "e2e-reflect"}
        for idx in range(4):
            await coordinator.after_action(
                {
                    "session": session,
                    "toolName": "exec" if idx % 2 else "read",
                    "ok": idx != 1,
                    "durationMs": 160 + idx * 20,
                    "retries": 2 if idx == 1 else 0,
                    "confidence": 0.9 if idx == 1 else 0.65,
                }
            )

        # Let queued after_action processing catch up.
        await asyncio.sleep(0.05)

        await coordinator.on_assistant_turn(
            {
                "session": session,
                "taskBoundary": True,
                "taskId": "task-e2e",
                "sessionSummary": {
                    "taskId": "task-e2e",
                    "actionCount": 4,
                    "successCount": 3,
                    "failureCount": 1,
                    "mixedOutcomes": True,
                    "timeSpentMs": 180000,
                },
            }
        )

        notification = coordinator.consume_reflection_notification({"session": session})
        self.assertIsNotNone(notification)
        self.assertIn("reflection", notification)
        self.assertIn("lesson_candidates", notification)
        self.assertGreaterEqual(len(notification["lesson_candidates"]), 1)

        stats = coordinator.stats()
        self.assertGreaterEqual(stats["lessons"]["success_rate"], 0.99)

        await coordinator.stop()


if __name__ == "__main__":
    unittest.main()
