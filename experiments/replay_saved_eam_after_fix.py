import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.rag.self_correcting.eam import EvidenceAssessmentModule
from Core.rag.self_correcting.controller import RetryController
from Core.rag.self_correcting.types import RetryState

# Workspace root that holds this repository (…/BookRAG/BookRAG).
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

PATH = str(
    WORKSPACE_ROOT
    / "test_output"
    / "controlled_manual"
    / "eval_single_query_sc_on_gbc_standard"
    / "query_001"
    / "retrieval_res.json"
)

QUERY = (
    "Which inlet protection device belongs to "
    "the circulation equipment serving Reactor R-2?"
)


with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

step = data["iterations"][0]

all_nodes = step["retrieval_nodes"]
old_assessments = step["self_correcting_assessments"]

eam = EvidenceAssessmentModule()
controller = RetryController()


for label, assessment_index in [
    ("INITIAL", 0),
    ("AFTER_S3", 1),
]:

    ids = {
        str(x)
        for x in old_assessments[assessment_index]["evidence_ids"]
    }

    nodes = [
        node
        for node in all_nodes
        if str(node.get("index_id")) in ids
    ]

    assessment = eam.assess(
        QUERY,
        nodes,
    )

    decision = controller.decide(
        assessment,
        RetryState(),
    )

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        "IDs =",
        sorted(
            ids,
            key=lambda x: int(x),
        ),
    )

    print(
        "STATUS =",
        assessment.status.value,
    )

    print(
        "DIAGNOSIS =",
        assessment.diagnosis.value,
    )

    print(
        "RELEVANCE =",
        round(
            assessment.relevance_score,
            3,
        ),
    )

    print(
        "COVERAGE =",
        round(
            assessment.coverage_score,
            3,
        ),
    )

    print(
        "SUPPORT =",
        round(
            assessment.support_score,
            3,
        ),
    )

    print(
        "CONTRADICTION =",
        round(
            assessment.contradiction_score,
            3,
        ),
    )

    print(
        "ACTION =",
        decision.action.value,
    )

    print(
        "STRATEGY =",
        (
            decision.strategy.value
            if decision.strategy
            else None
        ),
    )

    print(
        "STOP =",
        (
            decision.stop_reason.value
            if decision.stop_reason
            else None
        ),
    )