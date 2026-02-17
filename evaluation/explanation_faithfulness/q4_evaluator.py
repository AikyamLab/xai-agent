"""
Q4 Evaluator: Contrastive Instances (Why A != B)

Metric: 1 - Sim(F_A, F_B)
Higher score = better (features should be distinct between instances)

Similarity measures:
- Vision: Jaccard word similarity on concise feature phrases
- Text: Word overlap between extracted spans
- Tabular: Jaccard overlap weighted by rank agreement on top_features lists
"""

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
        try:
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
                    "features_a": features_a,
                    "features_b": features_b,
                    "similarity": similarity,
                    "threshold": 0.5,
                    "interpretation": "Higher score = more distinct features (better)"
                }
            )

        except Exception as e:
            return EvaluationResult(
                score=0.0,
                passed=False,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                errors=[str(e)]
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

        Extracts the actual text from each instance and computes word similarity.
        """
        try:
            start1, end1 = span1.get('start_index', 0), span1.get('end_index', 0)
            start2, end2 = span2.get('start_index', 0), span2.get('end_index', 0)

            if len(inputs) >= 2:
                text1 = inputs[0][start1:end1] if isinstance(inputs[0], str) else ""
                text2 = inputs[1][start2:end2] if isinstance(inputs[1], str) else ""
                return self._compute_text_similarity(text1, text2)

            # Same text fallback: character-level IoU
            intersection = max(0, min(end1, end2) - max(start1, start2))
            union = max(end1, end2) - min(start1, start2)
            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

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
        """
        try:
            list1 = features1.get('top_features', []) if isinstance(features1, dict) else []
            list2 = features2.get('top_features', []) if isinstance(features2, dict) else []

            # Fallback for single feature_key format
            if not list1 and isinstance(features1, dict) and 'feature_key' in features1:
                list1 = [features1['feature_key']]
            if not list2 and isinstance(features2, dict) and 'feature_key' in features2:
                list2 = [features2['feature_key']]

            if not list1 or not list2:
                return 0.0

            set1, set2 = set(list1), set(list2)
            intersection = set1 & set2
            union = set1 | set2

            jaccard = len(intersection) / len(union) if union else 0.0

            if not intersection:
                return 0.0

            # Rank agreement: average of 1/(1+|rank_diff|) for shared features
            rank_scores = []
            for feat in intersection:
                r1 = list1.index(feat) + 1
                r2 = list2.index(feat) + 1
                rank_scores.append(1.0 / (1.0 + abs(r1 - r2)))
            rank_agreement = sum(rank_scores) / len(rank_scores)

            return jaccard * rank_agreement

        except Exception:
            return 0.0
