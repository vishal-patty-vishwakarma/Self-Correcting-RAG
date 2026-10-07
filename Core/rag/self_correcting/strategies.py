from typing import Optional

from .types import (
    EvidenceAssessment,
    EvidenceDiagnosis,
    RecoveryStrategy,
    RetryState,
)


class RecoveryStrategySelector:
    """
    Map EAM diagnoses to recovery strategies.

    This class decides WHAT recovery strategy should be attempted.

    It does not itself perform retrieval.
    Actual BookRAG tree/KG expansion will be connected later.
    """

    def select(
        self,
        assessment: EvidenceAssessment,
        retry_state: RetryState,
    ) -> Optional[RecoveryStrategy]:
        """
        Select the next recovery strategy.

        Returns None when recovery should not continue.
        """

        if not retry_state.can_retry():
            return None

        diagnosis = assessment.diagnosis

        # No recovery required.
        if diagnosis == EvidenceDiagnosis.NONE:
            return None

        # Repeating the same evidence means another identical retry
        # is unlikely to help.
        if diagnosis == EvidenceDiagnosis.NO_PROGRESS:
            return self._fallback_strategy(retry_state)

        # Nearby context may contain omitted details, adjacent paragraphs,
        # sibling sections, parent sections, etc.
        if diagnosis == EvidenceDiagnosis.LOCAL_CONTEXT_GAP:
            return RecoveryStrategy.S1_LOCAL_CONTEXT

        # Missing named entities suggest using KG/GT-Link/entity occurrences.
        if diagnosis == EvidenceDiagnosis.MISSING_ENTITY:
            return RecoveryStrategy.S2_ENTITY_GUIDED

        # Explicit relationship-chain gap.
        if diagnosis == EvidenceDiagnosis.MULTI_HOP_GAP:
            return RecoveryStrategy.S3_GRAPH_MULTI_HOP

        # Current document scope appears too narrow.
        if diagnosis == EvidenceDiagnosis.STRUCTURAL_SCOPE_GAP:
            return RecoveryStrategy.S4_STRUCTURAL_SCOPE

        # Low relevance means the current retrieval neighborhood may simply
        # be wrong. Widen structural scope first.
        if diagnosis == EvidenceDiagnosis.LOW_RELEVANCE:
            return RecoveryStrategy.S4_STRUCTURAL_SCOPE

        # Relevant evidence exists but query coverage is incomplete.
        if diagnosis == EvidenceDiagnosis.LOW_COVERAGE:
            return RecoveryStrategy.S1_LOCAL_CONTEXT

        # Weak answer support often benefits from nearby corroborating context.
        if diagnosis == EvidenceDiagnosis.WEAK_SUPPORT:
            return RecoveryStrategy.S1_LOCAL_CONTEXT

        # Contradictory evidence requires careful widening of scope so that
        # superseding/revision context can be recovered.
        if diagnosis == EvidenceDiagnosis.CONTRADICTION:
            return RecoveryStrategy.S4_STRUCTURAL_SCOPE

        return None

    @staticmethod
    def _fallback_strategy(
        retry_state: RetryState,
    ) -> Optional[RecoveryStrategy]:
        """
        When the latest recovery produced no new evidence, avoid repeating
        exactly the same strategy.

        Escalation order:

            S1 -> S2 -> S3 -> S4

        This order is only a deterministic engineering baseline.
        Diagnosis-driven selection remains the primary mechanism.
        """

        escalation_order = [
            RecoveryStrategy.S1_LOCAL_CONTEXT,
            RecoveryStrategy.S2_ENTITY_GUIDED,
            RecoveryStrategy.S3_GRAPH_MULTI_HOP,
            RecoveryStrategy.S4_STRUCTURAL_SCOPE,
        ]

        used = set(retry_state.strategy_history)

        for strategy in escalation_order:
            if strategy not in used:
                return strategy

        return None