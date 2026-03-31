"""
Q4 Evaluator: Contrastive Instances (Why A != B)

Metric: 1 - Sim(F_A, F_B)
Higher score = better (features should be distinct between instances)

Similarity measures:
- Vision: Jaccard word similarity on concise feature phrases
- Text: Word overlap between extracted spans
- Tabular: Jaccard overlap weighted by rank agreement on top_features lists
"""

threshold = 0.3

from typing import Any, Dict, List, Set
import re

from ..base_evaluator import MultiInstanceEvaluator, EvaluationResult


class Q4Evaluator(MultiInstanceEvaluator):
    """Evaluator for Q4: Why instances A and B have different predictions"""

    @property
    def question_type(self) -> int:
        return 4

    @property
    def metric_name(self) -> str:
        return "Feature Distinctness"

    @property
    def metric_formula(self) -> str:
        return "1 - Sim(F_A, F_B)"

    def evaluate_multi(
        self,
        agent_output: Dict[str, Any],
        inputs: List[Any],
        model: Any,
        predictions: List[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q4: Compare feature similarity between instance A and B.

        Lower similarity = better explanation (features are distinct).

        Args:
            agent_output: Agent output with regions for input_A and input_B
            inputs: [input_A, input_B]
            model: Target model
            predictions: [prediction_A, prediction_B]
            **kwargs: Additional arguments

        Returns:
            EvaluationResult with 1-similarity score
        """
        output_data = agent_output.get('output', {})

        # Get features for both instances
        features_a = output_data.get('input_A')
        features_b = output_data.get('input_B')

        if features_a is None or features_b is None:
            return EvaluationResult(
                score=0.0,
                passed=False,
                errors=["Features not provided for both instances"]
            )

        # Validate agent output: bad responses → score 0.0
        if self.modality == "vision":
            # Q4 vision output is a concise feature phrase (string), not a bounding box
            for label, feat in [("input_A", features_a), ("input_B", features_b)]:
                text = str(feat).lower().strip()
                if not text or text in ('none', 'null', '{}', '[]'):
                    return EvaluationResult(score=0.0, passed=False,
                                            metric_name=self.metric_name,
                                            metric_formula=self.metric_formula,
                                            errors=[f"Empty vision phrase for {label}"])
                if any(m in text for m in self._REFUSAL_MARKERS):
                    return EvaluationResult(score=0.0, passed=False,
                                            metric_name=self.metric_name,
                                            metric_formula=self.metric_formula,
                                            errors=[f"Refusal marker in {label} vision phrase"])
        else:
            # text/tabular: features_a/b are dicts in standard region format
            inp_a = inputs[0] if inputs else None
            inp_b = inputs[1] if len(inputs) > 1 else None
            for label, feat, inp in [("input_A", features_a, inp_a),
                                      ("input_B", features_b, inp_b)]:
                err = self.validate_region(feat, inp, **kwargs)
                if err:
                    return EvaluationResult(score=0.0, passed=False,
                                            metric_name=self.metric_name,
                                            metric_formula=self.metric_formula,
                                            errors=[f"{label}: {err}"])

        # Calculate similarity based on modality
        if self.modality == "vision":
            similarity = self._compute_text_similarity(
                str(features_a),
                str(features_b)
            )
        elif self.modality == "text":
            similarity = self._compute_span_overlap(
                features_a,
                features_b,
                inputs
            )
        else:  # tabular
            similarity = self._compute_feature_similarity(
                features_a,
                features_b
            )

        # Score = 1 - similarity: higher = more distinct = better
        score = 1.0 - similarity

        # Passed if features are distinct enough
        passed = score > 0.5

        return EvaluationResult(
            score=score,
            passed=passed,
            metric_name=self.metric_name,
            metric_formula=self.metric_formula,
            details={
                "soft_score": score,
                "size_score": score,
                "features_a": features_a,
                "features_b": features_b,
                "similarity": similarity,
                "threshold": threshold,
                "interpretation": "Higher score = more distinct features (better)"
            }
        )

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute word-level similarity between two text descriptions.

        Uses Jaccard similarity on word sets.
        """
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into word set, filtering stop words."""
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'and', 'but', 'or',
                      'that', 'this', 'these', 'those', 'it', 'its'}
        return set(w for w in words if w not in stop_words)

    def _compute_span_overlap(
        self,
        span1: Dict,
        span2: Dict,
        inputs: List
    ) -> float:
        """
        Compute overlap between text spans from different instances.

        Expects new multi-span format: {"spans": [{"start_index": ..., "end_index": ...}]}.
        Extracts the actual text from each instance and computes word similarity.
        """
        spans1 = span1.get('spans')
        spans2 = span2.get('spans')
        if not spans1 or not spans2:
            return 1.0  # max similarity → score 0.0

        if len(inputs) < 2:
            return 1.0  # max similarity → score 0.0

        def extract_span_text(text, spans):
            if not isinstance(text, str):
                raise TypeError(f"Expected str input, got {type(text)}")
            return " ".join(text[s['start_index']:s['end_index']] for s in spans)

        text1 = extract_span_text(inputs[0], spans1)
        text2 = extract_span_text(inputs[1], spans2)
        return self._compute_text_similarity(text1, text2)

    @staticmethod
    def _strip_value_annotation(feat: str) -> str:
        """Strip parenthesized value annotations from feature strings.

        e.g. 'mean_compactness (0.03834-0.08468)' -> 'mean_compactness'
        """
        return re.sub(r'\s*\(.*?\)\s*$', '', str(feat)).strip()

    def _compute_feature_similarity(
        self,
        features1: Dict,
        features2: Dict
    ) -> float:
        """
        Compute similarity between tabular feature rankings.

        Uses Jaccard overlap weighted by rank agreement on shared features.
        - Jaccard measures how many features overlap between two top-k lists
        - Rank agreement measures whether shared features have similar ranks
        - Final similarity = Jaccard * rank_agreement (0 if no overlap)

        Feature strings may include value annotations like 'mean_compactness (0.123)'
        which are stripped before comparison so the same feature name matches.
        """
        if not isinstance(features1, dict) or not isinstance(features2, dict):
            return 1.0  # max similarity → score 0.0

        list1 = features1.get('feature_keys') or features1.get('top_features', [])
        list2 = features2.get('feature_keys') or features2.get('top_features', [])

        if not list1 or not list2:
            return 1.0  # max similarity → score 0.0

        # Normalize: strip value annotations before comparison
        norm1 = [self._strip_value_annotation(f) for f in list1]
        norm2 = [self._strip_value_annotation(f) for f in list2]

        set1, set2 = set(norm1), set(norm2)
        intersection = set1 & set2
        union = set1 | set2

        jaccard = len(intersection) / len(union) if union else 0.0

        if not intersection:
            return 0.0

        # Rank agreement: average of 1/(1+|rank_diff|) for shared features
        rank_scores = []
        for feat in intersection:
            r1 = norm1.index(feat) + 1
            r2 = norm2.index(feat) + 1
            rank_scores.append(1.0 / (1.0 + abs(r1 - r2)))
        rank_agreement = sum(rank_scores) / len(rank_scores)

        return jaccard * rank_agreement
