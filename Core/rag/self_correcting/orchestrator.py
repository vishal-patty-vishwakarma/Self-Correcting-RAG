from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from .controller import (
    RetryAction,
    RetryController,
    RetryDecision,
    RetryStopReason,
)
from .eam import EvidenceAssessmentModule
from .recovery import RecoveryExecutor
from .types import (
    EvidenceAssessment,
    RecoveryStrategy,
    RetryState,
)


@dataclass
class OrchestrationResult:
    final_nodes: List[Any]
    final_assessment: EvidenceAssessment
    retry_state: RetryState

    assessment_history: List[EvidenceAssessment] = field(
        default_factory=list
    )

    decision_history: List[RetryDecision] = field(
        default_factory=list
    )


class ClosedLoopOrchestrator:
    """
    Local closed-loop recovery orchestrator.

    Flow:

        evidence
           ↓
          EAM
           ↓
     Retry Controller
           ↓
     Recovery Executor
           ↓
      new evidence
           ↓
          EAM
           ↺

    Tree-based recovery is always anchored to the original retrieval
    nodes so that expansion nodes do not themselves trigger uncontrolled
    structural expansion.
    """

    def __init__(
        self,
        eam: EvidenceAssessmentModule,
        controller: RetryController,
        recovery_executor: RecoveryExecutor,
    ):
        self.eam = eam
        self.controller = controller
        self.recovery_executor = recovery_executor

    def run(
        self,
        query: str,
        initial_nodes: Sequence[Any],
        retry_state: Optional[RetryState] = None,
    ) -> OrchestrationResult:

        # --------------------------------------------------------------
        # Stable recovery anchors.
        #
        # S1/S4 should expand around the ORIGINAL BookRAG retrieval,
        # not around nodes produced by previous expansion strategies.
        # --------------------------------------------------------------

        recovery_anchors = list(initial_nodes)

        # --------------------------------------------------------------
        # Evidence accumulates across recovery attempts.
        # --------------------------------------------------------------

        current_nodes = list(initial_nodes)

        state = (
            retry_state
            if retry_state is not None
            else RetryState()
        )

        assessment_history: List[EvidenceAssessment] = []
        decision_history: List[RetryDecision] = []

        while True:

            # ==========================================================
            # 1. ASSESS
            # ==========================================================

            assessment = self.eam.assess(
                query=query,
                retrieval_nodes=current_nodes,
                previous_evidence_ids=state.previous_evidence_ids,
            )

            assessment_history.append(
                assessment
            )

            self.controller.update_current_evidence(
                retry_state=state,
                evidence_ids=assessment.evidence_ids,
            )

            # ==========================================================
            # 2. CONTROLLER
            # ==========================================================

            decision = self.controller.decide(
                assessment=assessment,
                retry_state=state,
            )

            # ==========================================================
            # 3. STOP
            # ==========================================================

            if decision.action == RetryAction.STOP:

                decision_history.append(
                    decision
                )

                return OrchestrationResult(
                    final_nodes=current_nodes,
                    final_assessment=assessment,
                    retry_state=state,
                    assessment_history=assessment_history,
                    decision_history=decision_history,
                )

            requested_strategy = decision.strategy

            # ==========================================================
            # 4. RESOLVE AGAINST REAL STRATEGY AVAILABILITY
            # ==========================================================

            actual_strategy = self._resolve_available_strategy(
                requested_strategy=requested_strategy,
                retry_state=state,
            )

            if actual_strategy is None:

                stop_decision = RetryDecision(
                    action=RetryAction.STOP,
                    strategy=None,
                    stop_reason=(
                        RetryStopReason.NO_APPLICABLE_STRATEGY
                    ),
                    attempt_count=state.attempt_count,
                    max_recovery_attempts=(
                        state.max_recovery_attempts
                    ),
                    rationale=(
                        "The requested recovery strategy is unavailable "
                        "and no unused executable fallback remains."
                    ),
                )

                decision_history.append(
                    stop_decision
                )

                return OrchestrationResult(
                    final_nodes=current_nodes,
                    final_assessment=assessment,
                    retry_state=state,
                    assessment_history=assessment_history,
                    decision_history=decision_history,
                )

            # ==========================================================
            # 5. RECORD ACTUAL EXECUTION DECISION
            # ==========================================================

            executed_decision = RetryDecision(
                action=RetryAction.CONTINUE,
                strategy=actual_strategy,
                stop_reason=None,
                attempt_count=state.attempt_count,
                max_recovery_attempts=(
                    state.max_recovery_attempts
                ),
                rationale=(
                    decision.rationale
                    + (
                        ""
                        if actual_strategy == requested_strategy
                        else (
                            f" Requested strategy "
                            f"{requested_strategy.value} is unavailable; "
                            f"executing {actual_strategy.value} instead."
                        )
                    )
                ),
            )

            decision_history.append(
                executed_decision
            )

            # ==========================================================
            # 6. REGISTER ATTEMPT
            # ==========================================================

            self.controller.register_recovery_attempt(
                retry_state=state,
                strategy=actual_strategy,
                evidence_ids_before_retry=assessment.evidence_ids,
            )

            # ==========================================================
            # 7. RECOVER FROM STABLE ORIGINAL ANCHORS
            # ==========================================================

            recovered_nodes = self.recovery_executor.execute(
                strategy=actual_strategy,
                current_nodes=recovery_anchors,
                query=query,
            )

            # ==========================================================
            # 8. ACCUMULATE NEW EVIDENCE WITHOUT DUPLICATES
            # ==========================================================

            current_nodes = self._merge_evidence(
                current_nodes,
                recovered_nodes,
            )

    # ==================================================================
    # STRATEGY AVAILABILITY
    # ==================================================================

    def _resolve_available_strategy(
        self,
        requested_strategy: Optional[RecoveryStrategy],
        retry_state: RetryState,
    ) -> Optional[RecoveryStrategy]:

        if (
            requested_strategy is not None
            and self.recovery_executor.is_available(
                requested_strategy
            )
            and requested_strategy
            not in retry_state.strategy_history
        ):
            return requested_strategy

        fallback_order = [
            RecoveryStrategy.S1_LOCAL_CONTEXT,
            RecoveryStrategy.S4_STRUCTURAL_SCOPE,
            RecoveryStrategy.S2_ENTITY_GUIDED,
            RecoveryStrategy.S3_GRAPH_MULTI_HOP,
        ]

        for strategy in fallback_order:

            if strategy in retry_state.strategy_history:
                continue

            if self.recovery_executor.is_available(
                strategy
            ):
                return strategy

        return None

    # ==================================================================
    # EVIDENCE MERGING
    # ==================================================================

    @classmethod
    def _merge_evidence(
        cls,
        existing: Sequence[Any],
        new_nodes: Sequence[Any],
    ) -> List[Any]:
        """
        Accumulate evidence without repeating the same BookRAG node ID.

        Existing evidence wins when duplicate IDs occur. This preserves
        the exact initial retrieval representation while adding genuinely
        new recovery nodes.
        """

        output: List[Any] = []
        seen = set()

        for node in list(existing) + list(new_nodes):

            key = cls._node_key(node)

            if key in seen:
                continue

            seen.add(key)
            output.append(node)

        return output

    @staticmethod
    def _node_key(node: Any) -> str:

        if isinstance(node, dict):
            node_id = node.get("index_id")

        else:
            node_id = getattr(
                node,
                "index_id",
                None,
            )

        if node_id is not None:
            return f"tree:{node_id}"

        # Fallback for unusual evidence objects without BookRAG IDs.
        return f"object:{id(node)}"