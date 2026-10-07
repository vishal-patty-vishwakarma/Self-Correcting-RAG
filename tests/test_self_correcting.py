import networkx as nx
import unittest
from pathlib import Path

from Core.Index.Tree import DocumentTree

from Core.rag.self_correcting import (
    ClosedLoopOrchestrator,
    EvidenceAssessment,
    EvidenceAssessmentModule,
    EvidenceDiagnosis,
    EvidenceStatus,
    RecoveryExecutor,
    RecoveryStrategy,
    RetryAction,
    RetryController,
    RetryState,
    RetryStopReason,
)


# Workspace root that holds this repository (…/BookRAG/BookRAG).
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

TREE_PATH = str(
    WORKSPACE_ROOT / "test_output" / "controlled_manual" / "tree.pkl"
)


class FakeGraphIndex:

    def __init__(self):

        self.kg = nx.Graph()

        cooling = "Cooling Loop C (SYSTEM)"
        pump = "Pump Y (EQUIPMENT)"
        strainer = "ST-C40 (STRAINER)"

        self.kg.add_node(
            cooling,
            entity_name="Cooling Loop C",
            entity_type="SYSTEM",
        )

        self.kg.add_node(
            pump,
            entity_name="Pump Y",
            entity_type="EQUIPMENT",
        )

        self.kg.add_node(
            strainer,
            entity_name="ST-C40",
            entity_type="STRAINER",
        )

        self.kg.add_edge(
            cooling,
            pump,
        )

        self.kg.add_edge(
            pump,
            strainer,
        )

        self.tree2kg = {
            37: {cooling},
            39: {pump},
            40: {strainer},
        }

        self.source_ids = {
            cooling: [37, 38],
            pump: [16, 39],
            strainer: [40],
        }

    def NodeName2TreeNodes(
        self,
        node_name,
    ):
        return list(
            self.source_ids.get(
                node_name,
                [],
            )
        )

class TestSelfCorrectingRAG(unittest.TestCase):
    def test_orchestrator_executes_s3_when_graph_available(self):
 
        fake_graph = FakeGraphIndex()

        recovery = RecoveryExecutor(
        tree=self.tree,
        graph_index=fake_graph,
        max_nodes=25,
        graph_hops=2,
        )

        orchestrator = ClosedLoopOrchestrator(
           eam=self.eam,
           controller=RetryController(),
           recovery_executor=recovery,
        )

        initial_nodes = [
           {
               "index_id": 37,
               "text": (
                   "Reactor R-2 is served by "
                   "Cooling Loop C."
                ),
            }
        ]

        result = orchestrator.run(
           query=(
               "What strainer protects the pump "
               "that serves Reactor R-2?"
            ),
            initial_nodes=initial_nodes,
        )

        self.assertGreaterEqual(
            result.retry_state.attempt_count,
            1,
        )

        self.assertEqual(
           result.retry_state.strategy_history[0],
           RecoveryStrategy.S3_GRAPH_MULTI_HOP,
        )

        self.assertEqual(
           result.decision_history[0].strategy,
           RecoveryStrategy.S3_GRAPH_MULTI_HOP,
        )

   
    def test_s2_entity_guided_recovery(self):

        fake_graph = FakeGraphIndex()

        recovery = RecoveryExecutor(
            tree=self.tree,
            graph_index=fake_graph,
            max_nodes=25,
        )

        self.assertTrue(
            recovery.is_available(
                RecoveryStrategy.S2_ENTITY_GUIDED
            )
        )

        nodes = recovery.execute(
            RecoveryStrategy.S2_ENTITY_GUIDED,
            [37],
            query="Where is Pump Y mentioned?",
        )

        ids = [
            node.index_id
            for node in nodes
        ]

        # Tree node 16 is another source occurrence
        # associated with Pump Y.
        self.assertIn(
            16,
            ids,
        )

        self.assertIn(
            39,
            ids,
        )

    def test_s3_graph_multihop_recovery(self):

        fake_graph = FakeGraphIndex()

        recovery = RecoveryExecutor(
            tree=self.tree,
            graph_index=fake_graph,
            max_nodes=25,
            graph_hops=2,
        )

        self.assertTrue(
            recovery.is_available(
                RecoveryStrategy.S3_GRAPH_MULTI_HOP
            )
        )

        nodes = recovery.execute(
            RecoveryStrategy.S3_GRAPH_MULTI_HOP,
            [37],
            query="What protects this circulation path?",
        )

        ids = [
            node.index_id
            for node in nodes
        ]

        # Graph path:
        #
        # Cooling Loop C
        #       ↓
        # Pump Y
        #       ↓
        # ST-C40
        #
        # ST-C40 maps back to tree node 40.
        self.assertIn(
            40,
            ids,
        )
    @classmethod
    def setUpClass(cls):
        cls.tree = DocumentTree.load_from_file(
            TREE_PATH
        )

        cls.eam = EvidenceAssessmentModule()

        cls.controller = RetryController()

        cls.recovery = RecoveryExecutor(
            cls.tree,
            max_nodes=25,
        )

        cls.orchestrator = ClosedLoopOrchestrator(
            cls.eam,
            cls.controller,
            cls.recovery,
        )

    # ==============================================================
    # EAM
    # ==============================================================

    def test_direct_fact_is_sufficient(self):

        assessment = self.eam.assess(
            "Who manufactures Valve X?",
            [
                {
                    "index_id": 13,
                    "text": (
                        "Valve X is manufactured by "
                        "Asterion Controls Ltd."
                    ),
                }
            ],
        )

        self.assertEqual(
            assessment.status,
            EvidenceStatus.SUFFICIENT,
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.NONE,
        )

    def test_multihop_gap_is_partial(self):

        assessment = self.eam.assess(
            (
                "What strainer protects the pump "
                "that serves Reactor R-2?"
            ),
            [
                {
                    "index_id": 37,
                    "text": (
                        "Reactor R-2 is served by "
                        "Cooling Loop C."
                    ),
                }
            ],
        )

        self.assertEqual(
            assessment.status,
            EvidenceStatus.PARTIAL,
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.MULTI_HOP_GAP,
        )

    def test_no_progress_detection(self):

        assessment = self.eam.assess(
            "Who manufactures Valve X?",
            [
                {
                    "index_id": 13,
                    "text": "Valve X inspection procedure.",
                }
            ],
            previous_evidence_ids=["13"],
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.NO_PROGRESS,
        )

        self.assertEqual(
            assessment.progress_score,
            0.0,
        )

    def test_empty_evidence_is_insufficient(self):

        assessment = self.eam.assess(
            "Who manufactures Valve X?",
            [],
        )

        self.assertEqual(
            assessment.status,
            EvidenceStatus.INSUFFICIENT,
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.LOW_RELEVANCE,
        )

    def test_unresolved_numeric_conflict(self):

        assessment = self.eam.assess(
            "What is the inspection interval for Valve X?",
            [
                {
                    "index_id": 900,
                    "text": (
                        "Valve X inspection interval "
                        "is 450 hours."
                    ),
                },
                {
                    "index_id": 901,
                    "text": (
                        "Valve X inspection interval "
                        "is 500 hours."
                    ),
                },
            ],
        )

        self.assertEqual(
            assessment.status,
            EvidenceStatus.CONTRADICTORY,
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.CONTRADICTION,
        )

        self.assertGreaterEqual(
            assessment.contradiction_score,
            0.60,
        )

    # ==============================================================
    # RETRY CONTROLLER
    # ==============================================================

    def test_sufficient_stops(self):

        assessment = EvidenceAssessment(
            status=EvidenceStatus.SUFFICIENT,
            relevance_score=1.0,
            coverage_score=1.0,
            support_score=0.9,
        )

        decision = self.controller.decide(
            assessment,
            RetryState(),
        )

        self.assertEqual(
            decision.action,
            RetryAction.STOP,
        )

        self.assertEqual(
            decision.stop_reason,
            RetryStopReason.SUFFICIENT,
        )

    def test_multihop_selects_s3(self):

        assessment = EvidenceAssessment(
            status=EvidenceStatus.PARTIAL,
            relevance_score=0.5,
            coverage_score=0.5,
            support_score=0.4,
            diagnosis=EvidenceDiagnosis.MULTI_HOP_GAP,
        )

        decision = self.controller.decide(
            assessment,
            RetryState(),
        )

        self.assertEqual(
            decision.strategy,
            RecoveryStrategy.S3_GRAPH_MULTI_HOP,
        )

    def test_max_attempts_stop(self):

        assessment = EvidenceAssessment(
            status=EvidenceStatus.PARTIAL,
            relevance_score=0.5,
            coverage_score=0.4,
            support_score=0.3,
            diagnosis=EvidenceDiagnosis.LOW_COVERAGE,
        )

        state = RetryState(
            attempt_count=4,
            max_recovery_attempts=4,
        )

        decision = self.controller.decide(
            assessment,
            state,
        )

        self.assertEqual(
            decision.action,
            RetryAction.STOP,
        )

        self.assertEqual(
            decision.stop_reason,
            RetryStopReason.MAX_ATTEMPTS,
        )

    def test_no_progress_escalates(self):

        assessment = EvidenceAssessment(
            status=EvidenceStatus.PARTIAL,
            relevance_score=0.7,
            coverage_score=0.4,
            support_score=0.4,
            progress_score=0.0,
            diagnosis=EvidenceDiagnosis.NO_PROGRESS,
        )

        state = RetryState(
            attempt_count=1,
            strategy_history=[
                RecoveryStrategy.S1_LOCAL_CONTEXT
            ],
        )

        decision = self.controller.decide(
            assessment,
            state,
        )

        self.assertEqual(
            decision.strategy,
            RecoveryStrategy.S2_ENTITY_GUIDED,
        )

    def test_unresolved_contradiction_selects_s4(self):

        assessment = EvidenceAssessment(
            status=EvidenceStatus.CONTRADICTORY,
            relevance_score=1.0,
            coverage_score=1.0,
            support_score=0.9,
            contradiction_score=0.75,
            diagnosis=EvidenceDiagnosis.CONTRADICTION,
        )

        decision = self.controller.decide(
            assessment,
            RetryState(),
        )

        self.assertEqual(
            decision.action,
            RetryAction.CONTINUE,
        )

        self.assertEqual(
            decision.strategy,
            RecoveryStrategy.S4_STRUCTURAL_SCOPE,
        )

    # ==============================================================
    # REAL BOOKRAG TREE RECOVERY
    # ==============================================================

    def test_s1_real_tree_recovery(self):

        nodes = self.recovery.execute(
            RecoveryStrategy.S1_LOCAL_CONTEXT,
            [37],
        )

        ids = [
            node.index_id
            for node in nodes
        ]

        self.assertEqual(
            ids,
            [37, 1, 38, 39, 40, 35, 41],
        )

    def test_s4_real_tree_recovery(self):

        nodes = self.recovery.execute(
            RecoveryStrategy.S4_STRUCTURAL_SCOPE,
            [37],
        )

        ids = [
            node.index_id
            for node in nodes
        ]

        self.assertEqual(
            ids,
            [
                37,
                1,
                38,
                39,
                40,
                33,
                34,
                35,
                36,
                41,
                42,
                43,
                44,
                45,
            ],
        )

    # ==============================================================
    # CONTROLLED DOCUMENT CASES
    # ==============================================================

    def test_controlled_absence_is_sufficient(self):

        nodes = [
            self.tree.get_node_by_index_id(44),
            self.tree.get_node_by_index_id(45),
        ]

        result = self.orchestrator.run(
            "What is the operating voltage of Valve X?",
            nodes,
        )

        self.assertEqual(
            result.final_assessment.status,
            EvidenceStatus.SUFFICIENT,
        )

        self.assertEqual(
            result.retry_state.attempt_count,
            0,
        )

    def test_resolved_supersession_is_sufficient(self):

        nodes = [
            self.tree.get_node_by_index_id(27),
            self.tree.get_node_by_index_id(43),
        ]

        result = self.orchestrator.run(
            "What is the current inspection interval for Valve X?",
            nodes,
        )

        self.assertEqual(
            result.final_assessment.status,
            EvidenceStatus.SUFFICIENT,
        )

        self.assertLess(
            result.final_assessment.contradiction_score,
            0.60,
        )

    # ==============================================================
    # CLOSED LOOP
    # ==============================================================

    def test_closed_loop_multihop_recovers_with_s1(self):

        initial_nodes = [
            {
                "index_id": 37,
                "text": (
                    "Reactor R-2 is served by "
                    "Cooling Loop C."
                ),
            }
        ]

        result = self.orchestrator.run(
            (
                "What strainer protects the pump "
                "that serves Reactor R-2?"
            ),
            initial_nodes,
        )

        self.assertEqual(
            result.final_assessment.status,
            EvidenceStatus.SUFFICIENT,
        )

        self.assertEqual(
            result.retry_state.attempt_count,
            1,
        )

        self.assertEqual(
            result.retry_state.strategy_history,
            [
                RecoveryStrategy.S1_LOCAL_CONTEXT
            ],
        )

        self.assertEqual(
            result.decision_history[-1].stop_reason,
            RetryStopReason.SUFFICIENT,
        )


    def test_unrelated_numeric_conflict_does_not_affect_strainer_query(self):

        assessment = self.eam.assess(
            (
                "What strainer protects the pump "
                "that serves Reactor R-2?"
            ),
            [
                {
                    "index_id": 1000,
                    "text": (
                        "Reactor R-2 is served by Cooling Loop C. "
                        "Pump Y is part of the cooling system."
                    ),
                },
                {
                    "index_id": 1001,
                    "text": (
                        "The historical Valve X inspection interval "
                        "was 450 hours."
                    ),
                },
                {
                    "index_id": 1002,
                    "text": (
                        "The current Valve X inspection interval "
                        "is 500 hours."
                    ),
                },
            ],
        )

        self.assertNotEqual(
            assessment.status,
            EvidenceStatus.CONTRADICTORY,
        )

        self.assertNotEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.CONTRADICTION,
        )

        self.assertLess(
            assessment.contradiction_score,
            0.60,
        )

    def test_unrelated_revision_markers_do_not_affect_voltage_query(self):

        assessment = self.eam.assess(
            "What is the operating voltage of Valve X?",
            [
                {
                    "index_id": 10,
                    "text": (
                        "No operating voltage is specified for "
                        "Valve X anywhere in this manual. "
                        "This omission is intentional."
                    ),
                },
                {
                    "index_id": 45,
                    "text": (
                        "This manual does not specify an operating "
                        "voltage for Valve X. "
                        "The value must not be inferred."
                    ),
                },
                {
                    "index_id": 27,
                    "text": (
                        "The inspection interval for Valve X in this "
                        "revision is 500 operating hours. "
                        "A historical interval of 450 hours is mentioned "
                        "only so that Service Bulletin SB-17 can explicitly "
                        "mark it as superseded."
                    ),
                },
                {
                    "index_id": 31,
                    "text": (
                        "Applicable equipment: VX-21 / Valve X. "
                        "Current actuator: AX-4. "
                        "Legacy actuator: AX-3 - no longer preferred "
                        "for new replacements."
                    ),
                },
            ],
        )

        self.assertEqual(
            assessment.status,
            EvidenceStatus.SUFFICIENT,
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.NONE,
        )

        self.assertLess(
            assessment.contradiction_score,
            0.60,
        )
    def test_conservative_term_normalization(self):

        self.assertTrue(
            self.eam._terms_match(
                "protection",
                "protected",
            )
        )

        self.assertTrue(
            self.eam._terms_match(
                "serving",
                "served",
            )
        )

        # Important: "serving" must NOT accidentally
        # match unrelated "service".
        self.assertFalse(
            self.eam._terms_match(
                "serving",
                "service",
            )
        )


    def test_paraphrased_multihop_evidence_is_sufficient(self):

        assessment = self.eam.assess(
            (
                "Which inlet protection device belongs to "
                "the circulation equipment serving Reactor R-2?"
            ),
            [
                {
                    "index_id": 2000,
                    "text": (
                        "Reactor R-2 uses circulation equipment "
                        "Pump Y / PX-40. Pump Y is protected by "
                        "the ST-C40 inlet strainer."
                    ),
                }
            ],
        )

        self.assertEqual(
            assessment.status,
            EvidenceStatus.SUFFICIENT,
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.NONE,
        )

        self.assertGreaterEqual(
            assessment.coverage_score,
            0.70,
        )

    # ==============================================================
    # ENTITY EXTRACTION: CAPITALIZED PREPOSITIONS
    # ==============================================================

    def test_capitalized_preposition_is_not_entity(self):

        assessment = self.eam.assess(
            (
                "For the circulation equipment serving "
                "Reactor R-2, what is its alternate "
                "identifier and which inlet strainer "
                "protects it?"
            ),
            [
                {
                    "index_id": 38,
                    "text": (
                        "Cooling Loop C removes process heat "
                        "from Reactor R-2. Pump Y (PX-40) "
                        "provides the circulation needed to "
                        "move coolant through the reactor "
                        "jacket and return header."
                    ),
                },
                {
                    "index_id": 40,
                    "text": (
                        "If coolant flow to Reactor R-2 is "
                        "lost, operators first verify Pump Y "
                        "availability, then inspect the "
                        "Cooling Loop C inlet strainer and "
                        "return header."
                    ),
                },
            ],
        )

        # "For" must not appear as a missing entity.
        self.assertNotIn(
            "For",
            assessment.missing_entities,
        )

    # ==============================================================
    # DIFFERENT-EQUIPMENT INTERVALS ARE NOT CONTRADICTORY
    # ==============================================================

    def test_different_equipment_intervals_not_contradictory(self):

        assessment = self.eam.assess(
            (
                "For the circulation equipment serving "
                "Reactor R-2, what is its alternate "
                "identifier and which inlet strainer "
                "protects it?"
            ),
            [
                {
                    "index_id": 38,
                    "text": (
                        "Cooling Loop C removes process heat "
                        "from Reactor R-2. Pump Y (PX-40) "
                        "provides the circulation needed."
                    ),
                },
                {
                    "index_id": 40,
                    "text": (
                        "If coolant flow to Reactor R-2 is "
                        "lost, operators first verify Pump Y "
                        "availability, then inspect the "
                        "Cooling Loop C inlet strainer."
                    ),
                },
                {
                    "index_id": 17,
                    "text": (
                        "Pump Y / PX-40. Routine inspection "
                        "interval: 750 operating hours."
                    ),
                },
                {
                    "index_id": 14,
                    "text": (
                        "Valve X / VX-21. Inspection "
                        "interval: 500 operating hours."
                    ),
                },
            ],
        )

        self.assertNotEqual(
            assessment.status,
            EvidenceStatus.CONTRADICTORY,
        )

        self.assertNotEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.CONTRADICTION,
        )

        self.assertLess(
            assessment.contradiction_score,
            0.60,
        )

    # ==============================================================
    # SAME-ATTRIBUTE NUMERIC CONFLICT REMAINS CONTRADICTORY
    # ==============================================================

    def test_same_attribute_conflict_still_contradictory(self):

        assessment = self.eam.assess(
            "What is the inspection interval for Valve X?",
            [
                {
                    "index_id": 900,
                    "text": (
                        "Valve X inspection interval "
                        "is 450 hours."
                    ),
                },
                {
                    "index_id": 901,
                    "text": (
                        "Valve X inspection interval "
                        "is 500 hours."
                    ),
                },
            ],
        )

        self.assertEqual(
            assessment.status,
            EvidenceStatus.CONTRADICTORY,
        )

        self.assertEqual(
            assessment.diagnosis,
            EvidenceDiagnosis.CONTRADICTION,
        )

        self.assertGreaterEqual(
            assessment.contradiction_score,
            0.60,
        )

    # ==============================================================
    # UNRELATED INTERVALS DO NOT AFFECT VOLTAGE QUERY
    # ==============================================================

    def test_unrelated_intervals_do_not_affect_voltage_query(self):

        assessment = self.eam.assess(
            "What is the operating voltage of Valve X?",
            [
                {
                    "index_id": 100,
                    "text": (
                        "Pump Y routine inspection interval "
                        "is 750 operating hours."
                    ),
                },
                {
                    "index_id": 101,
                    "text": (
                        "Valve X inspection interval "
                        "is 500 operating hours."
                    ),
                },
                {
                    "index_id": 44,
                    "text": (
                        "No operating voltage is specified "
                        "for Valve X in this manual."
                    ),
                },
            ],
        )

        self.assertNotEqual(
            assessment.status,
            EvidenceStatus.CONTRADICTORY,
        )

        self.assertLess(
            assessment.contradiction_score,
            0.60,
        )

    # ==============================================================
    # ENTITY EXTRACTION: LEGITIMATE IDENTIFIERS PRESERVED
    # ==============================================================

    def test_entity_extraction_preserves_identifiers(self):

        candidates = self.eam._guess_missing_entities(
            (
                "For the circulation equipment serving "
                "Reactor R-2, what is its alternate "
                "identifier and which inlet strainer "
                "protects it?"
            ),
            "Reactor R-2 is served by Cooling Loop C.",
        )

        # "For" must be filtered out.
        self.assertNotIn("For", candidates)

        # "Reactor" IS in the evidence, so it should NOT be missing.
        self.assertNotIn("Reactor", candidates)

        # "R-2" IS in the evidence, so it should NOT be missing.
        self.assertNotIn("R-2", candidates)


if __name__ == "__main__":
    unittest.main()