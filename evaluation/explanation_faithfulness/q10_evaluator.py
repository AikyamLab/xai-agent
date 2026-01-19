"""
Q10 Evaluator: Similar Instances, Different Predictions

Metric: -Sim(F_correct, F_wrong)
Lower similarity = better (features should be distinct)

Similarity measures:
- Vision: Word similarity on natural language descriptions
- Text: Word overlap between spans
- Tabular: Rank correlation of features
"""

from typing import Any, Dict, List, Set
import re

from ..base_evaluator import MultiInstanceEvaluator, EvaluationResult


class Q10Evaluator(MultiInstanceEvaluator):
    """Evaluator for Q10: Similar instances with different predictions"""

    @property
    def question_type(self) -> int:
        return 10

    @property
    def metric_name(self) -> str:
        return "Feature Distinctness"

    @property
    def metric_formula(self) -> str:
        return "-Sim(F_correct, F_wrong)"

    def evaluate_multi(
        self,
        agent_output: Dict[str, Any],
        inputs: List[Any],
        model: Any,
        predictions: List[Dict[str, Any]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate Q10: Compare feature similarity between correct and wrong instance.

        Lower similarity = better explanation (features are distinct).

        Args:
            agent_output: Agent output with correct/wrong instance features
            inputs: [correct_input, wrong_input]
            model: Target model
            predictions: [correct_prediction, wrong_prediction]
            **kwargs: Additional arguments

        Returns:
            EvaluationResult with negative similarity score
        """
        try:
            output_data = agent_output.get('output', {})

            # Get features for correct and wrong instances
            correct_features = output_data.get('correct_instance_features')
            wrong_features = output_data.get('wrong_instance_features')

            if correct_features is None or wrong_features is None:
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    errors=["Features not provided for both instances"]
                )

            # Calculate similarity based on modality
            if self.modality == "vision":
                similarity = self._compute_text_similarity(
                    str(correct_features),
                    str(wrong_features)
                )
            elif self.modality == "text":
                similarity = self._compute_span_overlap(
                    correct_features,
                    wrong_features,
                    inputs
                )
            else:  # tabular
                similarity = self._compute_feature_similarity(
                    correct_features,
                    wrong_features
                )

            # Score is negative similarity (lower similarity = higher score)
            score = -similarity

            # Passed if similarity is below threshold (features are distinct)
            passed = similarity < 0.5

            return EvaluationResult(
                score=score,
                passed=passed,
                metric_name=self.metric_name,
                metric_formula=self.metric_formula,
                details={
                    "correct_features": correct_features,
                    "wrong_features": wrong_features,
                    "similarity": similarity,
                    "threshold": 0.5,
                    "interpretation": "More negative = more distinct features (better)"
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
        # Tokenize and normalize
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into word set"""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        # Remove common stop words
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
        Compute overlap between text spans.

        Uses character-level IoU or word overlap.
        """
        try:
            start1, end1 = span1.get('start_index', 0), span1.get('end_index', 0)
            start2, end2 = span2.get('start_index', 0), span2.get('end_index', 0)

            # If spans are in different texts (different instances), compare words
            if len(inputs) >= 2:
                text1 = inputs[0][start1:end1] if isinstance(inputs[0], str) else ""
                text2 = inputs[1][start2:end2] if isinstance(inputs[1], str) else ""
                return self._compute_text_similarity(text1, text2)

            # Same text: compute IoU
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
        Compute similarity between tabular feature specifications.

        Uses exact match or rank correlation.
        """
        try:
            key1 = features1.get('feature_key', '')
            key2 = features2.get('feature_key', '')

            # Exact match: 1.0 similarity
            if key1 == key2:
                return 1.0

            # No match: 0.0 similarity
            return 0.0

        except Exception:
            return 0.0
