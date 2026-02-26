import unittest

from analysis.autonomy.communication import CommunicationPolicy


class CommunicationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = CommunicationPolicy(cooldown_turns=3)

    def test_high_confidence_silent(self):
        result = self.policy.evaluate(
            session_key="s1",
            directive="PROCEED",
            confidence=0.88,
            risk=0.2,
            risk_band="low",
            reason="High confidence",
            alternatives=[],
        )
        self.assertFalse(result.should_message)
        self.assertEqual(result.kind, "silent_high_confidence")

    def test_medium_confidence_low_risk_silent(self):
        result = self.policy.evaluate(
            session_key="s2",
            directive="PROCEED",
            confidence=0.64,
            risk=0.3,
            risk_band="low",
            reason="Manageable risk",
            alternatives=[],
        )
        self.assertFalse(result.should_message)
        self.assertEqual(result.kind, "silent_medium_confidence_low_risk")

    def test_medium_confidence_high_risk_one_line_caution(self):
        result = self.policy.evaluate(
            session_key="s3",
            directive="PROCEED_WITH_CAUTION",
            confidence=0.66,
            risk=0.62,
            risk_band="medium",
            reason="Medium confidence with elevated risk",
            alternatives=[],
        )
        self.assertTrue(result.should_message)
        self.assertIn("Caution:", result.message or "")
        self.assertTrue(result.noncritical_prompt)

    def test_low_confidence_suggests_alternatives(self):
        result = self.policy.evaluate(
            session_key="s4",
            directive="ASK_CLARIFICATION",
            confidence=0.42,
            risk=0.4,
            risk_band="low",
            reason="signals are mixed",
            alternatives=["A", "B", "C"],
        )
        self.assertTrue(result.should_message)
        self.assertIn("Confidence is low (42%)", result.message or "")
        self.assertIn("A or B", result.message or "")

    def test_critical_risk_requires_confirmation_even_during_cooldown(self):
        first = self.policy.evaluate(
            session_key="s5",
            directive="ASK_CLARIFICATION",
            confidence=0.40,
            risk=0.4,
            risk_band="low",
            reason="Need clarification",
            alternatives=[],
        )
        self.assertTrue(first.should_message)

        critical = self.policy.evaluate(
            session_key="s5",
            directive="ASK_CONFIRMATION",
            confidence=0.33,
            risk=0.95,
            risk_band="critical",
            reason="High risk",
            alternatives=[],
        )
        self.assertTrue(critical.should_message)
        self.assertTrue(critical.requires_confirmation)
        self.assertIn("Confirm before I proceed", critical.message or "")

    def test_cooldown_allows_only_one_noncritical_prompt_per_three_turns(self):
        one = self.policy.evaluate(
            session_key="s6",
            directive="ASK_CLARIFICATION",
            confidence=0.45,
            risk=0.5,
            risk_band="medium",
            reason="Low confidence step 1",
            alternatives=["A", "B"],
        )
        two = self.policy.evaluate(
            session_key="s6",
            directive="ASK_CLARIFICATION",
            confidence=0.44,
            risk=0.5,
            risk_band="medium",
            reason="Low confidence step 2",
            alternatives=["A", "B"],
        )
        three = self.policy.evaluate(
            session_key="s6",
            directive="ASK_CLARIFICATION",
            confidence=0.43,
            risk=0.5,
            risk_band="medium",
            reason="Low confidence step 3",
            alternatives=["A", "B"],
        )
        four = self.policy.evaluate(
            session_key="s6",
            directive="ASK_CLARIFICATION",
            confidence=0.42,
            risk=0.5,
            risk_band="medium",
            reason="Low confidence step 4",
            alternatives=["A", "B"],
        )

        self.assertTrue(one.should_message)
        self.assertFalse(two.should_message)
        self.assertFalse(three.should_message)
        self.assertTrue(four.should_message)

    def test_repeated_warning_collapses(self):
        first = self.policy.evaluate(
            session_key="s7",
            directive="PROCEED_WITH_CAUTION",
            confidence=0.60,
            risk=0.6,
            risk_band="medium",
            reason="Same caution reason",
            alternatives=[],
        )
        repeated = self.policy.evaluate(
            session_key="s7",
            directive="PROCEED_WITH_CAUTION",
            confidence=0.60,
            risk=0.6,
            risk_band="medium",
            reason="Same caution reason",
            alternatives=[],
            context={"turn": 4},
        )

        self.assertTrue(first.should_message)
        self.assertFalse(repeated.should_message)
        self.assertTrue(repeated.collapsed_repeat or repeated.suppressed_by_cooldown)


if __name__ == "__main__":
    unittest.main()
