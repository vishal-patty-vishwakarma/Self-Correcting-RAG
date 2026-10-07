import re
from difflib import SequenceMatcher
from typing import Any, Iterable, List, Optional, Sequence, Set
from .types import (
    EvidenceAssessment,
    EvidenceDiagnosis,
    EvidenceStatus,
)


class EvidenceAssessmentModule:
    """
    First deterministic implementation of the Evidence Assessment Module (EAM).

    Responsibilities:
    1. Inspect retrieved evidence.
    2. Estimate relevance, coverage, support, contradiction, and progress.
    3. Decide whether evidence is:
       - SUFFICIENT
       - PARTIAL
       - INSUFFICIENT
       - CONTRADICTORY
    4. Diagnose why recovery may be required.

    IMPORTANT:
    This is a heuristic baseline implementation.

    It does NOT yet use:
    - an LLM,
    - NLI,
    - embeddings,
    - a cross-encoder,
    - the BookRAG knowledge graph.

    The thresholds below are provisional engineering defaults and must later
    be calibrated experimentally.
    """

    def __init__(
        self,
        relevance_threshold: float = 0.25,
        sufficient_coverage_threshold: float = 0.70,
        sufficient_support_threshold: float = 0.65,
        contradiction_threshold: float = 0.60,
        progress_threshold: float = 0.10,
    ):
        self.relevance_threshold = relevance_threshold
        self.sufficient_coverage_threshold = sufficient_coverage_threshold
        self.sufficient_support_threshold = sufficient_support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.progress_threshold = progress_threshold

    def assess(
        self,
        query: str,
        retrieval_nodes: Sequence[Any],
        previous_evidence_ids: Optional[Sequence[str]] = None,
    ) -> EvidenceAssessment:
        """
        Assess currently retrieved evidence.

        retrieval_nodes may contain:
        - plain strings,
        - dictionaries,
        - BookRAG-like node objects.

        This loose interface is intentional so that the EAM can be unit-tested
        before being coupled to BookRAG's concrete node classes.
        """

        previous_evidence_ids = list(previous_evidence_ids or [])

        evidence_texts: List[str] = []
        evidence_ids: List[str] = []

        for index, node in enumerate(retrieval_nodes):
            text = self._extract_text(node)

            if text:
                evidence_texts.append(text)

            evidence_ids.append(
                self._extract_id(
                    node,
                    fallback_index=index,
                )
            )

        # --------------------------------------------------------------
        # CASE 1: retrieval returned no usable textual evidence
        # --------------------------------------------------------------

        if not evidence_texts:
            return EvidenceAssessment(
                status=EvidenceStatus.INSUFFICIENT,
                relevance_score=0.0,
                coverage_score=0.0,
                support_score=0.0,
                contradiction_score=0.0,
                progress_score=self._progress_score(
                    previous_evidence_ids,
                    evidence_ids,
                ),
                diagnosis=EvidenceDiagnosis.LOW_RELEVANCE,
                missing_content=[
                    "No usable evidence was retrieved."
                ],
                evidence_ids=evidence_ids,
                rationale=(
                    "No non-empty evidence text was available "
                    "for assessment."
                ),
            )

        # --------------------------------------------------------------
        # Convert query/evidence into lightweight lexical representations
        # --------------------------------------------------------------

        query_terms = self._content_terms(query)

        individual_evidence_terms = [
            self._content_terms(text)
            for text in evidence_texts
        ]

        combined_evidence_terms: Set[str] = set()

        for terms in individual_evidence_terms:
            combined_evidence_terms.update(terms)

        combined_evidence_text = " ".join(evidence_texts)

        # --------------------------------------------------------------
        # Compute first-pass assessment scores
        # --------------------------------------------------------------

        relevance_score = self._relevance_score(
            query_terms,
            individual_evidence_terms,
        )

        coverage_score = self._coverage_score(
            query_terms,
            combined_evidence_terms,
        )

        support_score = self._support_score(
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            evidence_count=len(evidence_texts),
        )

        contradiction_score = self._contradiction_score(
            query_terms=query_terms,
            evidence_texts=evidence_texts,
        )

        progress_score = self._progress_score(
            previous_evidence_ids,
            evidence_ids,
        )

        matched_query_terms = self._matched_query_terms(
            query_terms,
            combined_evidence_terms,
        )

        missing_terms = sorted(
            query_terms - matched_query_terms
        )

        missing_entities = self._guess_missing_entities(
            query,
            combined_evidence_text,
        )

        # --------------------------------------------------------------
        # Decide evidence status
        # --------------------------------------------------------------

        status = self._choose_status(
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            support_score=support_score,
            contradiction_score=contradiction_score,
        )

        # --------------------------------------------------------------
        # Diagnose why evidence is insufficient/partial
        # --------------------------------------------------------------

        diagnosis = self._diagnose(
            query=query,
            status=status,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            support_score=support_score,
            progress_score=progress_score,
            missing_entities=missing_entities,
        )

        missing_content = self._build_missing_content(
            diagnosis=diagnosis,
            missing_terms=missing_terms,
        )

        rationale = (
            f"Assessed {len(evidence_texts)} evidence item(s). "
            f"Relevance={relevance_score:.2f}, "
            f"coverage={coverage_score:.2f}, "
            f"support={support_score:.2f}, "
            f"contradiction={contradiction_score:.2f}, "
            f"progress={progress_score:.2f}."
        )

        return EvidenceAssessment(
            status=status,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            support_score=support_score,
            contradiction_score=contradiction_score,
            progress_score=progress_score,
            diagnosis=diagnosis,
            missing_entities=missing_entities,
            missing_content=missing_content,
            evidence_ids=evidence_ids,
            rationale=rationale,
        )

    # ==================================================================
    # STATUS DECISION
    # ==================================================================

    def _choose_status(
        self,
        relevance_score: float,
        coverage_score: float,
        support_score: float,
        contradiction_score: float,
    ) -> EvidenceStatus:

        if contradiction_score >= self.contradiction_threshold:
            return EvidenceStatus.CONTRADICTORY

        if (
            relevance_score >= self.relevance_threshold
            and coverage_score >= self.sufficient_coverage_threshold
            and support_score >= self.sufficient_support_threshold
        ):
            return EvidenceStatus.SUFFICIENT

        if relevance_score < self.relevance_threshold:
            return EvidenceStatus.INSUFFICIENT

        return EvidenceStatus.PARTIAL

    # ==================================================================
    # DIAGNOSIS
    # ==================================================================

    def _diagnose(
        self,
        query: str,
        status: EvidenceStatus,
        relevance_score: float,
        coverage_score: float,
        support_score: float,
        progress_score: float,
        missing_entities: Sequence[str],
    ) -> EvidenceDiagnosis:

        if status == EvidenceStatus.CONTRADICTORY:
            return EvidenceDiagnosis.CONTRADICTION

        if status == EvidenceStatus.SUFFICIENT:
            return EvidenceDiagnosis.NONE

        # A repeated retrieval that adds almost nothing should stop/escalate.
        if progress_score < self.progress_threshold:
            return EvidenceDiagnosis.NO_PROGRESS

        if relevance_score < self.relevance_threshold:
            return EvidenceDiagnosis.LOW_RELEVANCE

        # If a named entity from the query never appeared in evidence,
        # entity-guided recovery may later be useful.
        if missing_entities:
            return EvidenceDiagnosis.MISSING_ENTITY

        # Detect questions that probably require relationship traversal.
        if self._looks_multihop(query):
            return EvidenceDiagnosis.MULTI_HOP_GAP

        if coverage_score < self.sufficient_coverage_threshold:
            return EvidenceDiagnosis.LOW_COVERAGE

        if support_score < self.sufficient_support_threshold:
            return EvidenceDiagnosis.WEAK_SUPPORT

        return EvidenceDiagnosis.LOW_COVERAGE
    @staticmethod
    def _term_variants(
        term: str,
    ) -> Set[str]:
        """
        Return conservative morphological variants.

        This remains dependency-free and intentionally avoids
        aggressive stemming.

        Examples:

            protection -> protect
            protected  -> protect
            serving    -> serve
            served     -> serve
            circulation -> circulat
            circulating -> circulat

        It must NOT collapse unrelated words such as:

            serving != service
        """

        variants = {
            term,
        }

        if len(term) < 4:
            return variants

        # --------------------------------------------------------------
        # -ing forms
        #
        # serving -> serv / serve
        # operating -> operat / operate
        # --------------------------------------------------------------

        if (
            term.endswith("ing")
            and len(term) > 5
        ):
            stem = term[:-3]

            variants.add(stem)
            variants.add(
                stem + "e"
            )

        # --------------------------------------------------------------
        # -ed forms
        #
        # protected -> protect
        # served -> serv / serve
        # --------------------------------------------------------------

        if (
            term.endswith("ed")
            and len(term) > 4
        ):
            stem = term[:-2]

            variants.add(stem)
            variants.add(
                stem + "e"
            )

        # --------------------------------------------------------------
        # Nominalized -ion forms.
        #
        # protection -> protect
        # operation -> operat
        # --------------------------------------------------------------

        if (
            term.endswith("ion")
            and len(term) > 6
        ):
            variants.add(
                term[:-3]
            )

        # --------------------------------------------------------------
        # Simple plural / third-person forms.
        #
        # serves -> serve
        # devices -> device
        # protects -> protect
        # --------------------------------------------------------------

        if (
            term.endswith("es")
            and len(term) > 4
        ):
            variants.add(
                term[:-1]
            )

            variants.add(
                term[:-2]
            )

        elif (
            term.endswith("s")
            and len(term) > 4
        ):
            variants.add(
                term[:-1]
            )

        return {
            variant
            for variant in variants
            if len(variant) >= 4
        }
    # ==================================================================
    # TERM MATCHING
    # ==================================================================
    @staticmethod
    def _terms_match(
        query_term: str,
        evidence_term: str,
    ) -> bool:
        """
        Conservative lexical + morphological term matching.
        """

        if query_term == evidence_term:
            return True

        query_variants = (
            EvidenceAssessmentModule._term_variants(
                query_term
            )
        )

        evidence_variants = (
            EvidenceAssessmentModule._term_variants(
                evidence_term
            )
        )

        if (
            query_variants
            & evidence_variants
        ):
            return True

        if min(
            len(query_term),
            len(evidence_term),
        ) < 4:
            return False

        similarity = SequenceMatcher(
            None,
            query_term,
            evidence_term,
        ).ratio()

        return similarity >= 0.82
    def _matched_query_terms(
        self,
        query_terms: Set[str],
        evidence_terms: Set[str],
    ) -> Set[str]:
        """
        Return query terms represented by the evidence,
        allowing conservative lexical and morphological matching.
        """

        matched: Set[str] = set()

        for query_term in query_terms:
            for evidence_term in evidence_terms:
                if self._terms_match(
                    query_term,
                    evidence_term,
                ):
                    matched.add(query_term)
                    break

        return matched
    # ==================================================================
    # SCORE FUNCTIONS
    # ==================================================================

    def _relevance_score(
        self,
        query_terms: Set[str],
        evidence_term_sets: Sequence[Set[str]],
    ) -> float:
        """
        Relevance asks:

            "Is at least one retrieved evidence item strongly related
             to this query?"
        """

        if not query_terms or not evidence_term_sets:
            return 0.0

        scores = []

        for terms in evidence_term_sets:
            matched = self._matched_query_terms(
                query_terms,
                terms,
            )

            scores.append(
                len(matched) / len(query_terms)
            )

        return min(
            1.0,
            max(scores, default=0.0),
        )

    def _coverage_score(
        self,
        query_terms: Set[str],
        combined_evidence_terms: Set[str],
    ) -> float:
        """
        Coverage asks:

            "Across ALL retrieved evidence, how much of the query
             is represented?"
        """

        if not query_terms:
            return 0.0

        matched = self._matched_query_terms(
            query_terms,
            combined_evidence_terms,
        )

        return min(
            1.0,
            len(matched) / len(query_terms),
        )

    @staticmethod
    def _support_score(
        relevance_score: float,
        coverage_score: float,
        evidence_count: int,
    ) -> float:
        """
        Initial proxy for answer support.

        Future versions should replace/augment this with an NLI or
        cross-encoder verifier.

        For now:
        - relevance contributes 45%
        - coverage contributes 45%
        - multiple evidence items can contribute a small bonus
        """

        evidence_bonus = min(
            0.10,
            max(0, evidence_count - 1) * 0.02,
        )

        score = (
            0.45 * relevance_score
            + 0.45 * coverage_score
            + evidence_bonus
        )

        return min(
            1.0,
            max(0.0, score),
        )

    # ==================================================================
    # CONTRADICTION
    # ==================================================================

    def _contradiction_score(
        self,
        query_terms: Set[str],
        evidence_texts: Sequence[str],
        ) -> float:
        """
        Deterministic, query-conditioned contradiction heuristic.

        Detects:
        1. explicit conflict/revision language;
        2. conflicting measurement claims.

        IMPORTANT:
        Contradictions are considered only when the relevant evidence
        is related to the current query.

        Example:

            Query:
                "What is the inspection interval for Valve X?"

            Evidence:
                "450 hours"
                "500 hours"

            -> relevant contradiction

        But:

            Query:
                "What strainer protects the pump serving Reactor R-2?"

            Unrelated maintenance evidence:
                "450 hours"
                "500 hours"

            -> NOT a contradiction for this query

        This remains a deterministic v0 heuristic, not semantic NLI.
        """

        contradiction_markers = {
            "however",
            "superseded",
            "obsolete",
            "no longer",
            "incorrect",
            "conflicts with",
        }

        # --------------------------------------------------------------
        # 1. Determine which evidence items are relevant to THIS query.
        # --------------------------------------------------------------

        evidence_term_sets = [
            self._content_terms(text)
            for text in evidence_texts
        ]

        relevant_indices = []

        for index, terms in enumerate(
            evidence_term_sets
        ):

            matched = self._matched_query_terms(
                query_terms,
                terms,
            )

            match_ratio = (
                len(matched)
                / max(len(query_terms), 1)
            )

            # Keep an evidence item contradiction-relevant when:
            #
            # - at least two meaningful query terms match; OR
            # - at least half of a short query is represented.
            #
            # This prevents unrelated recovered sections from creating
            # false contradictions.
            if (
                len(matched) >= 2
                or match_ratio >= 0.50
            ):
                relevant_indices.append(
                    index
                )

        # No query-relevant evidence means there cannot be a
        # query-relevant contradiction.
        if not relevant_indices:
            return 0.0

        relevant_texts = [
            evidence_texts[index]
            for index in relevant_indices
        ]

        relevant_term_sets = [
            evidence_term_sets[index]
            for index in relevant_indices
        ]

        # --------------------------------------------------------------
        # 2. Explicit contradiction/revision markers.
        #
        # Marker evidence needs STRONG query overlap.
        #
        # The broader relevant_indices filter is intentionally suitable
        # for general evidence assessment, but it is too permissive for
        # contradiction markers.
        #
        # Example:
        #
        # Query:
        #   "What is the operating voltage of Valve X?"
        #
        # Evidence:
        #   "Valve X inspection guidance was superseded."
        #   "AX-3 is no longer preferred."
        #
        # Those statements share the entity Valve X, but they do not
        # contradict the queried operating-voltage attribute.
        #
        # Therefore revision/conflict markers are allowed to contribute
        # only from evidence with stronger query-term coverage.
        # --------------------------------------------------------------

        marker_relevant_texts = []

        for text, terms in zip(
            relevant_texts,
            relevant_term_sets,
        ):
            matched = self._matched_query_terms(
                query_terms,
                terms,
            )

            match_ratio = (
                len(matched)
                / max(len(query_terms), 1)
            )

            # Stronger requirement than the general evidence-relevance
            # filter used above.
            #
            # This prevents entity-only or loosely related evidence from
            # creating a contradiction for a different queried attribute.
            if match_ratio > 0.75:
                marker_relevant_texts.append(text)

        combined = " ".join(
            marker_relevant_texts
        ).lower()

        marker_hits = sum(
            marker in combined
            for marker in contradiction_markers
        )

        marker_score = 0.0

        if marker_hits == 1:
            marker_score = 0.35

        elif marker_hits == 2:
            marker_score = 0.60

        elif marker_hits >= 3:
            marker_score = min(
                1.0,
                0.60
                + 0.10 * (marker_hits - 2),
            )

        # --------------------------------------------------------------
        # 3. Query-relevant unresolved measurement conflicts.
        #
        # Numbers are considered only when attached to known units.
        #
        # This prevents identifiers such as:
        #
        #   VX-21
        #   SB-17
        #   AX-4
        #
        # from becoming numeric contradictions.
        # --------------------------------------------------------------

        numeric_conflict_score = 0.0

        def extract_measurements(
            text: str,
        ):
            """
            Return measurements grouped by normalized unit.

            Examples:

                500 operating hours
                24 nominal volts
            """

            pattern = re.compile(
                r"(?<![A-Za-z0-9-])"
                r"(\d+(?:\.\d+)?)"
                r"\s+"
                r"(?:[A-Za-z]+\s+){0,2}"
                r"(hours?|hrs?|volts?|"
                r"psi|bar|rpm|"
                r"minutes?|mins?|seconds?|secs?|"
                r"mm|cm|kg|"
                r"liters?|litres?|"
                r"percent|%)"
                r"\b",
                re.IGNORECASE,
            )

            measurements = {}

            for value, unit in pattern.findall(
                text
            ):

                unit = unit.lower()

                unit_map = {
                    "hour": "hours",
                    "hr": "hours",
                    "hrs": "hours",
                    "volt": "volts",
                    "minute": "minutes",
                    "min": "minutes",
                    "mins": "minutes",
                    "second": "seconds",
                    "sec": "seconds",
                    "secs": "seconds",
                    "liter": "liters",
                    "litre": "liters",
                    "litres": "liters",
                }

                unit = unit_map.get(
                    unit,
                    unit,
                )

                measurements.setdefault(
                    unit,
                    set(),
                ).add(value)

            return measurements

        # Work only with evidence already proven relevant to the query.
        measurement_sets = [
            extract_measurements(text)
            for text in relevant_texts
        ]

        if len(relevant_texts) >= 2:

            for i in range(
                len(relevant_texts)
            ):

                for j in range(
                    i + 1,
                    len(relevant_texts),
                ):

                    # Require the two evidence items themselves to
                    # describe substantially overlapping subject matter.
                    # Use semantic terms (no HTML structural tokens)
                    # so that table markup does not make unrelated
                    # equipment entries appear related.
                    shared_terms = (
                        self._semantic_terms(
                            relevant_texts[i]
                        )
                        & self._semantic_terms(
                            relevant_texts[j]
                        )
                    )

                    if len(shared_terms) < 2:
                        continue

                    shared_units = (
                        set(measurement_sets[i])
                        & set(measurement_sets[j])
                    )

                    for unit in shared_units:

                        values_i = (
                            measurement_sets[i][unit]
                        )

                        values_j = (
                            measurement_sets[j][unit]
                        )

                        # Genuine unresolved conflict:
                        #
                        # evidence A -> 450 hours
                        # evidence B -> 500 hours
                        #
                        # with no shared value.
                        if (
                            values_i
                            and values_j
                            and values_i.isdisjoint(
                                values_j
                            )
                        ):
                            # Additional relevance gate:
                            # Only treat this as a contradiction
                            # when both measurement contexts are
                            # related to the query's information
                            # need.  This prevents unrelated
                            # maintenance intervals, voltages,
                            # etc. from creating false
                            # contradictions when the query asks
                            # about a different attribute.
                            ctx_i = self._measurement_context_relevance(
                                query_terms,
                                relevant_texts[i],
                            )
                            ctx_j = self._measurement_context_relevance(
                                query_terms,
                                relevant_texts[j],
                            )

                            if ctx_i and ctx_j:
                                numeric_conflict_score = 0.75
                                break

                    if numeric_conflict_score > 0:
                        break

                if numeric_conflict_score > 0:
                    break

        return max(
            marker_score,
            numeric_conflict_score,
        )


    # ==================================================================
    # PROGRESS
    # ==================================================================

    @staticmethod
    def _progress_score(
        previous_evidence_ids: Sequence[str],
        current_evidence_ids: Sequence[str],
    ) -> float:
        """
        Estimate whether recovery produced new evidence.

        No previous evidence means this is the initial retrieval,
        so progress defaults to 1.0.
        """

        if not previous_evidence_ids:
            return 1.0

        previous = set(previous_evidence_ids)
        current = set(current_evidence_ids)

        if not current:
            return 0.0

        new_items = current - previous

        return min(
            1.0,
            len(new_items) / len(current),
        )

    # ==================================================================
    # NODE ADAPTER HELPERS
    # ==================================================================

    @staticmethod
    def _extract_text(node: Any) -> str:
        """
        Extract text from strings, dictionaries, or BookRAG-like objects.
        """

        if node is None:
            return ""

        if isinstance(node, str):
            return node.strip()

        if isinstance(node, dict):

            parts = []

            for key in (
                "text",
                "content",
                "table_body",
                "caption",
                "footnote",
                "summary",
                "node_text",
                "page_content",
                "description",
            ):
                value = node.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                    and value.strip() not in parts
                ):
                    parts.append(value.strip())

            if parts:
                return "\n".join(parts)

            return str(node).strip()
        # BookRAG TreeNode stores its actual document text
        # inside node.meta_info.content.
        meta_info = getattr(node, "meta_info", None)

        if meta_info is not None:

            parts = []

            content = getattr(
                meta_info,
                "content",
                None,
            )

            table_body = getattr(
                meta_info,
                "table_body",
                None,
            )

            caption = getattr(
                meta_info,
                "caption",
                None,
            )

            footnote = getattr(
                meta_info,
                "footnote",
                None,
            )

            for value in (
                content,
                table_body,
                caption,
                footnote,
            ):
                if (
                    isinstance(value, str)
                    and value.strip()
                ):
                    parts.append(
                        value.strip()
                    )

            if parts:
                return "\n".join(parts)
            return str(node).strip()

        for attr in (
            "text",
            "content",
            "summary",
            "node_text",
            "page_content",
            "description",
        ):
            value = getattr(node, attr, None)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return str(node).strip()

    @staticmethod
    def _extract_id(
        node: Any,
        fallback_index: int,
    ) -> str:

        if isinstance(node, dict):
            for key in (
                "index_id",
                "node_idx",
                "id",
                "node_id",
            ):
                if key in node and node[key] is not None:
                    return str(node[key])

        for attr in (
            "index_id",
            "node_idx",
            "id",
            "node_id",
        ):
            value = getattr(node, attr, None)

            if value is not None:
                return str(value)

        return f"evidence_{fallback_index}"

    # ==================================================================
    # TEXT HELPERS
    # ==================================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:

        return re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]*",
            text.lower(),
        )

    def _content_terms(
        self,
        text: str,
    ) -> Set[str]:

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "what",
            "which",
            "who",
            "with",
            "belong",
            "belongs",
            "belonging",
        }

        return {
            token
            for token in self._tokenize(text)
            if token not in stopwords
        }

    # ==================================================================
    # STRUCTURAL TOKEN FILTER
    # ==================================================================

    # HTML / table markup tokens that should not count as meaningful
    # semantic content when determining whether two evidence items
    # share substantially overlapping subject matter.
    _STRUCTURAL_TOKENS = frozenset({
        "table",
        "tr",
        "td",
        "th",
        "tbody",
        "thead",
        "tfoot",
        "col",
        "colgroup",
    })

    @classmethod
    def _semantic_terms(
        cls,
        text: str,
    ) -> Set[str]:
        """
        Content terms with structural markup tokens removed.

        Used for contradiction relevance checks where table structure
        should not make unrelated equipment entries appear related.
        """

        stopwords = {
            "a", "an", "and", "are", "as", "at",
            "be", "by", "for", "from", "how", "in",
            "is", "it", "of", "on", "or", "that",
            "the", "this", "to", "what", "which",
            "who", "with", "belong", "belongs",
            "belonging",
        }

        return {
            token
            for token in re.findall(
                r"[A-Za-z0-9][A-Za-z0-9_-]*",
                text.lower(),
            )
            if token not in stopwords
            and token not in cls._STRUCTURAL_TOKENS
        }

    # ==================================================================
    # MEASUREMENT CONTEXT RELEVANCE
    # ==================================================================

    @staticmethod
    def _measurement_context_relevance(
        query_terms: Set[str],
        measurement_text: str,
    ) -> bool:
        """
        Check whether the local context surrounding a numeric
        measurement is related to the query's information need.

        This prevents unrelated measurements (e.g. inspection
        intervals for different equipment) from creating false
        contradictions when the query asks about a different
        attribute (e.g. alternate identifier or inlet strainer).

        A measurement context is considered relevant when at least
        two non-structural query terms appear in the measurement's
        surrounding text.
        """

        context_tokens = set(
            re.findall(
                r"[a-z][a-z0-9_-]*",
                measurement_text.lower(),
            )
        )

        relevant_query = query_terms - {
            "table", "tr", "td", "th",
        }

        matched = sum(
            1 for term in relevant_query
            if term in context_tokens
        )

        return matched >= 2

    # ==================================================================
    # ENTITY GAP
    # ==================================================================

    @staticmethod
    def _guess_missing_entities(
        query: str,
        evidence_text: str,
    ) -> List[str]:
        """
        Simple initial entity-gap heuristic.

        Finds capitalized or identifier-like strings in the query and
        checks whether they occur in the evidence.

        Example:
            Reactor R-2
            Valve X
            VX-21
        """

        candidates = re.findall(
            r"\b(?:"
            r"[A-Z][A-Za-z0-9_-]*"
            r"|[A-Z]{2,}[A-Za-z0-9_-]*"
            r"|[A-Za-z]+-\d+[A-Za-z0-9_-]*"
            r")\b",
            query,
        )

        ignored_question_words = {
            "Who",
            "What",
            "Which",
            "Where",
            "When",
            "Why",
            "How",
        }

        # Common function words that may appear capitalized at the
        # start of a sentence but are not named entities.
        ignored_function_words = {
            "For",
            "The",
            "A",
            "An",
            "In",
            "On",
            "By",
            "From",
            "To",
            "With",
            "At",
            "Of",
            "And",
            "Or",
            "But",
            "If",
            "As",
            "Is",
            "It",
            "Its",
            "Be",
            "Are",
            "Was",
            "Were",
            "Has",
            "Had",
            "Have",
            "Not",
            "This",
            "That",
            "These",
            "Those",
        }

        candidates = [
            candidate
            for candidate in candidates
            if candidate not in ignored_question_words
            and candidate not in ignored_function_words
        ]

        evidence_lower = evidence_text.lower()

        missing: List[str] = []

        for candidate in candidates:
            if (
                candidate.lower() not in evidence_lower
                and candidate not in missing
            ):
                missing.append(candidate)

        return missing

    # ==================================================================
    # MULTI-HOP SIGNAL
    # ==================================================================

    @staticmethod
    def _looks_multihop(query: str) -> bool:
        """
        Detect wording that often implies a relationship chain.

        This is only an initial heuristic.
        """

        q = query.lower()

        signals = (
            "that serves",
            "which serves",
            "connected to",
            "through",
            "associated with",
            "belongs to",
            "used by",
            "feeds",
            "supplies",
            "protects the",
        )

        return any(
            signal in q
            for signal in signals
        )

    # ==================================================================
    # HUMAN-READABLE MISSING CONTENT
    # ==================================================================

    @staticmethod
    def _build_missing_content(
        diagnosis: EvidenceDiagnosis,
        missing_terms: Iterable[str],
    ) -> List[str]:

        terms = list(missing_terms)

        if diagnosis == EvidenceDiagnosis.LOW_RELEVANCE:
            return [
                "More query-relevant evidence is required."
            ]

        if diagnosis == EvidenceDiagnosis.MISSING_ENTITY:
            return [
                "One or more query entities are absent from the evidence."
            ]

        if diagnosis == EvidenceDiagnosis.MULTI_HOP_GAP:
            return [
                "The evidence appears to be missing one or more "
                "relationships required to complete the query."
            ]

        if diagnosis == EvidenceDiagnosis.LOW_COVERAGE:
            if terms:
                return [
                    "Evidence does not yet cover query terms/concepts: "
                    + ", ".join(terms[:8])
                ]

            return [
                "Relevant evidence is incomplete."
            ]

        if diagnosis == EvidenceDiagnosis.WEAK_SUPPORT:
            return [
                "Evidence is relevant but does not strongly support an answer."
            ]

        if diagnosis == EvidenceDiagnosis.CONTRADICTION:
            return [
                "Retrieved evidence contains possible conflicting claims."
            ]

        if diagnosis == EvidenceDiagnosis.NO_PROGRESS:
            return [
                "The latest retrieval attempt added little or no new evidence."
            ]

        return []
