import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.rag.self_correcting.eam import EvidenceAssessmentModule
from Core.rag.self_correcting.controller import RetryController
from Core.rag.self_correcting.types import RetryState


# ----------------------------------------------------------------------
# Change these only when screening another saved baseline run.
# ----------------------------------------------------------------------

RETRIEVAL_FILE = (
    PROJECT_ROOT.parent.parent
    / "test_output"
    / "controlled_manual"
    / "eval_single_query_sc_off_gbc_standard"
    / "query_001"
    / "retrieval_res.json"
)


def find_first_iteration(data):
    """
    Return the first BookRAG iteration object.

    Current BookRAG output normally stores it under:
        data["iterations"][0]

    A small fallback is included so the script fails clearly
    if the JSON structure changes.
    """

    if isinstance(data, dict):

        iterations = data.get("iterations")

        if isinstance(iterations, list) and iterations:
            return iterations[0]

    raise RuntimeError(
        "Could not locate iterations[0] in retrieval_res.json"
    )


def extract_query(data, iteration):
    """
    Try several known/common locations for the original query.
    """

    candidates = []

    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("question"),
                data.get("query"),
            ]
        )

    if isinstance(iteration, dict):
        candidates.extend(
            [
                iteration.get("question"),
                iteration.get("query"),
                iteration.get("sub_question"),
            ]
        )

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def normalize_node(node):
    """
    Convert BookRAG retrieval-node JSON into the minimal shape
    expected by EvidenceAssessmentModule.
    """

    if not isinstance(node, dict):
        return None

    node_id = node.get("index_id")

    if node_id is None:
        node_id = node.get("id")

    text_parts = []

    for key in (
        "text",
        "content",
        "table_body",
    ):
        value = node.get(key)

        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())

    text = "\n".join(text_parts)

    if not text:
        return None

    return {
        "index_id": node_id,
        "text": text,
    }


def extract_retrieval_nodes(iteration):
    """
    Read baseline retrieval evidence from the first iteration.
    """

    raw_nodes = iteration.get("retrieval_nodes", [])

    normalized = []

    for node in raw_nodes:

        item = normalize_node(node)

        if item is not None:
            normalized.append(item)

    return normalized


def main():

    print("=" * 80)
    print("BASELINE EVIDENCE -> EAM SCREEN")
    print("=" * 80)

    print()
    print("Retrieval file:")
    print(RETRIEVAL_FILE)

    if not RETRIEVAL_FILE.exists():
        raise FileNotFoundError(
            f"Retrieval file does not exist:\n{RETRIEVAL_FILE}"
        )

    with open(
        RETRIEVAL_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    iteration = find_first_iteration(data)

    query = extract_query(
        data,
        iteration,
    )

    if query is None:
        print()
        print("ERROR:")
        print(
            "The query could not be recovered automatically "
            "from retrieval_res.json."
        )
        print()
        print(
            "Add it manually to the QUERY_OVERRIDE variable "
            "or send me the JSON structure."
        )
        return

    nodes = extract_retrieval_nodes(
        iteration,
    )

    print()
    print("Query:")
    print(query)

    print()
    print(
        "Retrieved evidence IDs:",
        [
            node.get("index_id")
            for node in nodes
        ],
    )

    print(
        "Evidence count:",
        len(nodes),
    )

    # ------------------------------------------------------------------
    # EAM
    # ------------------------------------------------------------------

    eam = EvidenceAssessmentModule()

    assessment = eam.assess(
        query,
        nodes,
    )

    print()
    print("-" * 80)
    print("EAM ASSESSMENT")
    print("-" * 80)

    print(
        "Status:",
        assessment.status.value,
    )

    print(
        "Diagnosis:",
        assessment.diagnosis.value,
    )

    print(
        "Relevance:",
        round(
            assessment.relevance_score,
            3,
        ),
    )

    print(
        "Coverage:",
        round(
            assessment.coverage_score,
            3,
        ),
    )

    print(
        "Support:",
        round(
            assessment.support_score,
            3,
        ),
    )

    print(
        "Contradiction:",
        round(
            assessment.contradiction_score,
            3,
        ),
    )

    print(
        "Progress:",
        round(
            assessment.progress_score,
            3,
        ),
    )

    print(
        "Missing entities:",
        assessment.missing_entities,
    )

    print(
        "Missing content:",
        assessment.missing_content,
    )

    print()
    print(
        "Rationale:",
        assessment.rationale,
    )

    # ------------------------------------------------------------------
    # Retry Controller
    # ------------------------------------------------------------------

    controller = RetryController()

    state = RetryState()

    decision = controller.decide(
        assessment,
        state,
    )

    print()
    print("-" * 80)
    print("CONTROLLER DECISION")
    print("-" * 80)

    print(
        "Action:",
        decision.action.value,
    )

    print(
        "Strategy:",
        (
            decision.strategy.value
            if decision.strategy is not None
            else None
        ),
    )

    print(
        "Stop reason:",
        (
            decision.stop_reason.value
            if decision.stop_reason is not None
            else None
        ),
    )

    print(
        "Attempt count:",
        decision.attempt_count,
    )

    print()
    print(
        "Rationale:",
        decision.rationale,
    )

    # ------------------------------------------------------------------
    # Simple screening verdict
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SCREENING VERDICT")
    print("=" * 80)

    if decision.strategy is not None:

        print(
            "PROMISING RECOVERY CASE"
        )

        print(
            "Baseline evidence would trigger:",
            decision.strategy.value,
        )

        print()
        print(
            "This query is worth running with "
            "Self-Correcting RAG."
        )

    else:

        print(
            "NO RECOVERY TRIGGERED"
        )

        print()
        print(
            "EAM already considers the baseline evidence "
            "sufficient, so this is not a useful recovery case."
        )


if __name__ == "__main__":
    main()