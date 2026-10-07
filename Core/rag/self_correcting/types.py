from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceStatus(str, Enum):
    """
    Overall evidence state returned by the Evidence Assessment Module.
    """

    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTORY = "CONTRADICTORY"


class EvidenceDiagnosis(str, Enum):
    """
    Reason why the currently retrieved evidence is or is not adequate.

    The EAM diagnoses the problem.
    It does NOT directly choose how retrieval should recover.
    """

    NONE = "NONE"

    # Retrieved evidence is mostly unrelated to the query.
    LOW_RELEVANCE = "LOW_RELEVANCE"

    # Evidence is relevant, but important parts of the query are unanswered.
    LOW_COVERAGE = "LOW_COVERAGE"

    # Evidence mentions the right topic/entities but does not support the answer.
    WEAK_SUPPORT = "WEAK_SUPPORT"

    # Important entities or concepts from the query are missing.
    MISSING_ENTITY = "MISSING_ENTITY"

    # Required information appears to require another relation/hop.
    MULTI_HOP_GAP = "MULTI_HOP_GAP"

    # Relevant information probably exists nearby in document structure.
    LOCAL_CONTEXT_GAP = "LOCAL_CONTEXT_GAP"

    # Current structural scope appears too narrow.
    STRUCTURAL_SCOPE_GAP = "STRUCTURAL_SCOPE_GAP"

    # Retrieved evidence contains conflicting claims.
    CONTRADICTION = "CONTRADICTION"

    # Recovery produced no useful new evidence.
    NO_PROGRESS = "NO_PROGRESS"


class RecoveryStrategy(str, Enum):
    """
    Recovery strategies available to the Retry Controller.

    The controller/strategy selector will use these later.
    """

    S1_LOCAL_CONTEXT = "S1_LOCAL_CONTEXT"
    S2_ENTITY_GUIDED = "S2_ENTITY_GUIDED"
    S3_GRAPH_MULTI_HOP = "S3_GRAPH_MULTI_HOP"
    S4_STRUCTURAL_SCOPE = "S4_STRUCTURAL_SCOPE"


class EvidenceAssessment(BaseModel):
    """
    Structured output of the Evidence Assessment Module.

    Scores are intentionally normalized to [0, 1].
    Thresholds will be calibrated experimentally later.
    """

    status: EvidenceStatus

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant the retrieved evidence is to the query.",
    )

    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How much of the query is covered by the evidence.",
    )

    support_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the evidence supports an answer.",
    )

    contradiction_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated amount of contradiction in the evidence.",
    )

    progress_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "How much useful evidence was gained compared with the "
            "previous retrieval attempt."
        ),
    )

    diagnosis: EvidenceDiagnosis = EvidenceDiagnosis.NONE

    missing_entities: List[str] = Field(default_factory=list)

    missing_content: List[str] = Field(default_factory=list)

    evidence_ids: List[str] = Field(default_factory=list)

    rationale: str = ""


class RetryState(BaseModel):
    """
    Runtime state maintained across recovery attempts.

    This prevents uncontrolled retrieval loops and lets the controller
    determine whether recovery is still making progress.
    """

    attempt_count: int = Field(default=0, ge=0)

    max_recovery_attempts: int = Field(default=4, ge=1)

    strategy_history: List[RecoveryStrategy] = Field(default_factory=list)

    previous_evidence_ids: List[str] = Field(default_factory=list)

    current_evidence_ids: List[str] = Field(default_factory=list)

    previous_status: Optional[EvidenceStatus] = None

    previous_coverage_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    previous_support_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    no_progress_count: int = Field(default=0, ge=0)

    def can_retry(self) -> bool:
        """
        Return True while the hard recovery-attempt limit has not been reached.
        """
        return self.attempt_count < self.max_recovery_attempts

    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """
        Record one recovery attempt.
        """
        self.strategy_history.append(strategy)
        self.attempt_count += 1