from enum import Enum
from typing import List, Optional, Sequence

from pydantic import BaseModel

from .strategies import RecoveryStrategySelector
from .types import (
    EvidenceAssessment,
    EvidenceDiagnosis,
    EvidenceStatus,
    RecoveryStrategy,
    RetryState,
)


class RetryAction(str, Enum):
    """
    High-level action returned by the Retry Controller.
    """

    CONTINUE = "CONTINUE"
    STOP = "STOP"


class RetryStopReason(str, Enum):
    """
    Explicit reason why the closed-loop recovery process stopped.
    """

    SUFFICIENT = "SUFFICIENT"
    MAX_ATTEMPTS = "MAX_ATTEMPTS"
    NO_APPLICABLE_STRATEGY = "NO_APPLICABLE_STRATEGY"
    NO_PROGRESS = "NO_PROGRESS"
    CONTRADICTION_UNRESOLVED = "CONTRADICTION_UNRESOLVED"


class RetryDecision(BaseModel):
    """
    Output of one Retry Controller decision.
    """

    action: RetryAction

    strategy: Optional[RecoveryStrategy] = None

    stop_reason: Optional[RetryStopReason] = None

    attempt_count: int

    max_recovery_attempts: int

    rationale: str = ""


class RetryController:
    """
    Bounded controller for Self-Correcting RAG.

    Responsibilities:
    1. Inspect the latest EvidenceAssessment.
    2. Decide STOP or CONTINUE.
    3. Select the next recovery strategy.
    4. Prevent repeated-strategy loops.
    5. Enforce the maximum recovery-attempt limit.
    6. Track evidence state across retries.

    The controller does NOT:
    - perform retrieval,
    - generate an answer,
    - assess evidence itself.

    Those responsibilities remain separated.
    """

    def __init__(
        self,
        selector: Optional[RecoveryStrategySelector] = None,
    ):
        self.selector = (
            selector
            if selector is not None
            else RecoveryStrategySelector()
        )

    # ==================================================================
    # MAIN DECISION
    # ==================================================================

    def decide(
        self,
        assessment: EvidenceAssessment,
        retry_state: RetryState,
    ) -> RetryDecision:
        """
        Decide whether recovery should stop or continue.
        """

        # --------------------------------------------------------------
        # STOP RULE 1:
        # Evidence is already sufficient.
        # --------------------------------------------------------------

        if assessment.status == EvidenceStatus.SUFFICIENT:
            return RetryDecision(
                action=RetryAction.STOP,
                stop_reason=RetryStopReason.SUFFICIENT,
                attempt_count=retry_state.attempt_count,
                max_recovery_attempts=retry_state.max_recovery_attempts,
                rationale=(
                    "Evidence is sufficient. "
                    "No additional retrieval is required."
                ),
            )

        # --------------------------------------------------------------
        # STOP RULE 2:
        # Hard maximum recovery-attempt limit.
        # --------------------------------------------------------------

        if not retry_state.can_retry():
            return RetryDecision(
                action=RetryAction.STOP,
                stop_reason=RetryStopReason.MAX_ATTEMPTS,
                attempt_count=retry_state.attempt_count,
                max_recovery_attempts=retry_state.max_recovery_attempts,
                rationale=(
                    "Maximum number of recovery attempts has been reached."
                ),
            )

        # --------------------------------------------------------------
        # CONTRADICTION HANDLING
        #
        # First contradiction:
        #   allow S4 structural expansion to search for revision,
        #   superseding, or wider contextual evidence.
        #
        # If S4 was already tried and contradiction remains:
        #   stop rather than looping.
        # --------------------------------------------------------------

        if assessment.status == EvidenceStatus.CONTRADICTORY:
            if (
                RecoveryStrategy.S4_STRUCTURAL_SCOPE
                in retry_state.strategy_history
            ):
                return RetryDecision(
                    action=RetryAction.STOP,
                    stop_reason=RetryStopReason.CONTRADICTION_UNRESOLVED,
                    attempt_count=retry_state.attempt_count,
                    max_recovery_attempts=retry_state.max_recovery_attempts,
                    rationale=(
                        "Contradiction remains after structural scope "
                        "expansion. Recovery is stopped to avoid "
                        "unsupported conflict resolution."
                    ),
                )

            return RetryDecision(
                action=RetryAction.CONTINUE,
                strategy=RecoveryStrategy.S4_STRUCTURAL_SCOPE,
                attempt_count=retry_state.attempt_count,
                max_recovery_attempts=retry_state.max_recovery_attempts,
                rationale=(
                    "Contradictory evidence detected. "
                    "Widen structural scope once to search for "
                    "superseding or clarifying context."
                ),
            )

        # --------------------------------------------------------------
        # Ask the strategy selector for the diagnosis-driven strategy.
        # --------------------------------------------------------------

        strategy = self.selector.select(
            assessment=assessment,
            retry_state=retry_state,
        )

        # --------------------------------------------------------------
        # STOP RULE 3:
        # No strategy is applicable.
        # --------------------------------------------------------------

        if strategy is None:
            stop_reason = (
                RetryStopReason.NO_PROGRESS
                if assessment.diagnosis
                == EvidenceDiagnosis.NO_PROGRESS
                else RetryStopReason.NO_APPLICABLE_STRATEGY
            )

            return RetryDecision(
                action=RetryAction.STOP,
                stop_reason=stop_reason,
                attempt_count=retry_state.attempt_count,
                max_recovery_attempts=retry_state.max_recovery_attempts,
                rationale=(
                    "No unused recovery strategy is applicable "
                    "to the current evidence diagnosis."
                ),
            )

        # --------------------------------------------------------------
        # LOOP PROTECTION:
        #
        # If diagnosis-driven selection proposes a strategy that has
        # already been attempted, escalate to another unused strategy.
        # --------------------------------------------------------------

        if strategy in retry_state.strategy_history:
            strategy = self._next_unused_strategy(
                retry_state
            )

            if strategy is None:
                stop_reason = (
                    RetryStopReason.NO_PROGRESS
                    if assessment.diagnosis
                    == EvidenceDiagnosis.NO_PROGRESS
                    else RetryStopReason.NO_APPLICABLE_STRATEGY
                )

                return RetryDecision(
                    action=RetryAction.STOP,
                    stop_reason=stop_reason,
                    attempt_count=retry_state.attempt_count,
                    max_recovery_attempts=retry_state.max_recovery_attempts,
                    rationale=(
                        "The recommended recovery strategy was already "
                        "used and no unused strategy remains."
                    ),
                )

        # --------------------------------------------------------------
        # CONTINUE:
        # A valid unused recovery strategy exists.
        # --------------------------------------------------------------

        return RetryDecision(
            action=RetryAction.CONTINUE,
            strategy=strategy,
            attempt_count=retry_state.attempt_count,
            max_recovery_attempts=retry_state.max_recovery_attempts,
            rationale=(
                f"Evidence status is {assessment.status.value} "
                f"with diagnosis {assessment.diagnosis.value}. "
                f"Recovery strategy {strategy.value} selected."
            ),
        )

    # ==================================================================
    # STATE TRANSITIONS
    # ==================================================================

    @staticmethod
    def register_recovery_attempt(
        retry_state: RetryState,
        strategy: RecoveryStrategy,
        evidence_ids_before_retry: Sequence[str],
    ) -> RetryState:
        """
        Record a recovery attempt immediately before executing retrieval.

        Example:

            Current evidence = [37]

            choose S3

            register_recovery_attempt(...)

            previous_evidence_ids = [37]
            attempt_count += 1
            strategy_history += S3

            recovery retrieves [37, 38, 39]

            EAM can then compare:
                previous = [37]
                current  = [37, 38, 39]
        """

        retry_state.previous_evidence_ids = list(
            evidence_ids_before_retry
        )

        retry_state.register_strategy(strategy)

        return retry_state

    @staticmethod
    def update_current_evidence(
        retry_state: RetryState,
        evidence_ids: Sequence[str],
    ) -> RetryState:
        """
        Store evidence returned by the latest retrieval/recovery step.
        """

        retry_state.current_evidence_ids = list(
            evidence_ids
        )

        return retry_state

    # ==================================================================
    # FALLBACK / ESCALATION
    # ==================================================================

    @staticmethod
    def _next_unused_strategy(
        retry_state: RetryState,
    ) -> Optional[RecoveryStrategy]:
        """
        Return the next unused recovery strategy.

        Deterministic fallback order:

            S1 → S2 → S3 → S4

        This is loop protection, not the primary selection mechanism.
        """

        escalation_order: List[RecoveryStrategy] = [
            RecoveryStrategy.S1_LOCAL_CONTEXT,
            RecoveryStrategy.S2_ENTITY_GUIDED,
            RecoveryStrategy.S3_GRAPH_MULTI_HOP,
            RecoveryStrategy.S4_STRUCTURAL_SCOPE,
        ]

        used = set(
            retry_state.strategy_history
        )

        for strategy in escalation_order:
            if strategy not in used:
                return strategy

        return None