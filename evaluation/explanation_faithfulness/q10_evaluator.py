"""
Q10 Evaluator: Similar Instances, Different Predictions

Metric: -Sim(F_correct, F_wrong)
Lower similarity = better (features should be distinct)

Similarity measures:
- Vision: Word similarity on natural language descriptions
- Text: Word overlap between spans extracted from input texts
- Tabular: Feature key overlap
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
            inputs: List of input data (strings, dicts, or tensors)
            model: Target model
            predictions: List of prediction dicts
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

            # Soft score: 1 - similarity, range [0, 1], higher = more distinct
            score = 1.0 - similarity

            # Passed logic unchanged: similarity below threshold
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
                    "interpretation": "Soft score: 1 - similarity, range [0,1], higher = more distinct"
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

    def _extract_text(self, inp: Any) -> str:
        """Extract plain text from an input that may be a string, dict, or other type."""
        if isinstance(inp, str):
            return inp
        if isinstance(inp, dict):
            # NLI format: combine premise + hypothesis
            if 'premise' in inp:
                return f"{inp.get('premise', '')} {inp.get('hypothesis', '')}"
            if 'text' in inp:
                return inp['text']
            if 'review_text' in inp:
                return inp['review_text']
        return str(inp) if inp else ""

    def _compute_span_overlap(
        self,
        feat1: Dict,
        feat2: Dict,
        inputs: List
    ) -> float:
        """
        Compute overlap between text features from correct/wrong instances.

        Strategy:
        1. If both features have valid start_index/end_index AND we can extract
           text from inputs, slice the spans and compare word overlap.
        2. Otherwise, fall back to comparing description/span text fields
           using word-level Jaccard similarity.
        """
        try:
            start1 = feat1.get('start_index')
            end1 = feat1.get('end_index')
            start2 = feat2.get('start_index')
            end2 = feat2.get('end_index')

            has_spans = (
                start1 is not None and end1 is not None and end1 > start1 and
                start2 is not None and end2 is not None and end2 > start2
            )

            if has_spans and len(inputs) >= 2:
                # Extract text from inputs (handles str, NLI dict, etc.)
                full_text1 = self._extract_text(inputs[0])
                full_text2 = self._extract_text(inputs[1])

                if full_text1 and full_text2:
                    span_text1 = full_text1[start1:end1]
                    span_text2 = full_text2[start2:end2]
                    if span_text1 and span_text2:
                        return self._compute_text_similarity(span_text1, span_text2)

            # Fallback: compare description or span text fields directly
            desc1 = feat1.get('description', feat1.get('span', ''))
            desc2 = feat2.get('description', feat2.get('span', ''))
            if desc1 and desc2:
                return self._compute_text_similarity(str(desc1), str(desc2))

            return 0.0

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
