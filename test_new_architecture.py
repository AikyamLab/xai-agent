"""
Test script for new modular architecture

Tests:
1. Import all modules
2. Create question templates
3. Create agents
4. Validate output schemas
5. Test evaluators (without model)
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)

    errors = []

    # Test prompts module
    try:
        from prompts import (
            PromptBuilder,
            QuestionCategory,
            Modality,
            get_prompt_builder,
            Q1MostResponsiblePromptBuilder,
            Q2LeastResponsiblePromptBuilder,
            Q3DistinctivePromptBuilder,
            Q4ContrastiveInstancesPromptBuilder,
            Q5MaskPredictionPromptBuilder,
            Q6FlipPredictionPromptBuilder,
            Q7ChangePredictionPromptBuilder,
            Q8IrrelevantPartsPromptBuilder,
            Q9SharedFeaturePromptBuilder,
            Q10SimilarDifferentPromptBuilder,
        )
        print("  [OK] prompts module")
    except ImportError as e:
        errors.append(f"prompts: {e}")
        print(f"  [FAIL] prompts module: {e}")

    # Test output schemas
    try:
        from prompts.output_schemas import (
            get_output_schema,
            validate_output,
            QUESTION_OUTPUT_SCHEMAS,
        )
        print("  [OK] output_schemas")
    except ImportError as e:
        errors.append(f"output_schemas: {e}")
        print(f"  [FAIL] output_schemas: {e}")

    # Test agents module
    try:
        from agents import (
            BaseAgent,
            ProposerAgent,
            ActorAgent,
            CriticAgent,
        )
        print("  [OK] agents module")
    except ImportError as e:
        errors.append(f"agents: {e}")
        print(f"  [FAIL] agents module: {e}")

    # Test evaluation module
    try:
        from evaluation import (
            BaseEvaluator,
            EvaluationResult,
            get_evaluator,
            get_masker,
            MaskingStrategy,
        )
        print("  [OK] evaluation module")
    except ImportError as e:
        errors.append(f"evaluation: {e}")
        print(f"  [FAIL] evaluation module: {e}")

    # Test evaluators
    try:
        from evaluation.explanation_faithfulness import (
            Q1Evaluator,
            Q2Evaluator,
            Q3Evaluator,
            Q4Evaluator,
            Q5Evaluator,
            Q6Evaluator,
            Q7Evaluator,
            Q8Evaluator,
            Q9Evaluator,
            Q10Evaluator,
        )
        print("  [OK] explanation_faithfulness evaluators")
    except ImportError as e:
        errors.append(f"evaluators: {e}")
        print(f"  [FAIL] evaluators: {e}")

    # Test strategy_faithfulness placeholder
    try:
        from evaluation.strategy_faithfulness import (
            StrategyFaithfulnessEvaluator,
            ToolCompatibilityChecker,
        )
        print("  [OK] strategy_faithfulness (placeholder)")
    except ImportError as e:
        errors.append(f"strategy_faithfulness: {e}")
        print(f"  [FAIL] strategy_faithfulness: {e}")

    # Test question_templates_new
    try:
        from question_templates_new import (
            get_question_template,
            get_prompt_builder,
            QuestionTemplate,
            QUESTION_METADATA,
        )
        print("  [OK] question_templates_new")
    except ImportError as e:
        errors.append(f"question_templates_new: {e}")
        print(f"  [FAIL] question_templates_new: {e}")

    # Test three_agent_system_new
    try:
        from three_agent_system_new import (
            create_three_agent_system,
            ThreeAgentPipeline,
        )
        print("  [OK] three_agent_system_new")
    except ImportError as e:
        errors.append(f"three_agent_system_new: {e}")
        print(f"  [FAIL] three_agent_system_new: {e}")

    if errors:
        print(f"\n  {len(errors)} import error(s) detected")
        return False
    else:
        print(f"\n  All imports successful!")
        return True


def test_question_templates():
    """Test question template creation"""
    print("\n" + "=" * 70)
    print("TEST 2: Question Templates")
    print("=" * 70)

    try:
        from question_templates_new import get_question_template, QUESTION_METADATA

        modalities = ["vision", "text", "tabular"]

        for q_type in range(1, 11):
            for modality in modalities:
                try:
                    template = get_question_template(q_type, modality)
                    assert template.q_type == q_type
                    assert template.modality == modality
                    assert template.template is not None
                    assert template.prompt_builder is not None
                except Exception as e:
                    print(f"  [FAIL] Q{q_type} ({modality}): {e}")
                    return False

        print(f"  [OK] All 10 question types x 3 modalities = 30 templates created")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_output_schemas():
    """Test output schema validation"""
    print("\n" + "=" * 70)
    print("TEST 3: Output Schema Validation")
    print("=" * 70)

    try:
        from prompts.output_schemas import get_output_schema, validate_output

        # Test Q1 vision schema
        schema = get_output_schema(1, "vision")
        assert "output" in schema
        assert "bounding_box" in schema["output"]
        print("  [OK] Q1 vision schema has bounding_box")

        # Test Q5 schema
        schema = get_output_schema(5, "vision")
        assert "prediction_changes" in schema["output"]
        print("  [OK] Q5 schema has prediction_changes")

        # Test Q6 schema
        schema = get_output_schema(6, "text")
        assert "change_plan" in schema["output"]
        print("  [OK] Q6 text schema has change_plan")

        # Test validation
        valid_output = {
            "output": {"bounding_box": [10, 20, 100, 150]},
            "explanation": "Test",
            "confidence": 0.85
        }
        is_valid, errors = validate_output(valid_output, 1, "vision")
        assert is_valid, f"Validation failed: {errors}"
        print("  [OK] Valid output passes validation")

        # Test invalid output
        invalid_output = {"output": {}}
        is_valid, errors = validate_output(invalid_output, 1, "vision")
        assert not is_valid, "Invalid output should fail validation"
        print("  [OK] Invalid output fails validation correctly")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_prompt_builders():
    """Test prompt builder functionality"""
    print("\n" + "=" * 70)
    print("TEST 4: Prompt Builders")
    print("=" * 70)

    try:
        from prompts import get_prompt_builder

        # Test context
        context = {
            "user_question": "Which part of the input was most responsible?",
            "model_info": {
                "model_name": "resnet18",
                "architecture": "ResNet",
                "num_classes": 10
            },
            "prediction": {
                "predicted_class": 5,
                "predicted_class_idx": 5,
                "confidence": 0.92
            }
        }

        # Test Q1 proposer prompt
        builder = get_prompt_builder(1, "vision")
        proposer_prompt = builder.build_proposer_prompt(context)
        assert len(proposer_prompt) > 100
        assert "JSON" in proposer_prompt
        print(f"  [OK] Q1 proposer prompt generated ({len(proposer_prompt)} chars)")

        # Test Q1 actor prompt
        strategy = {"strategy_type": "tools", "selected_tools": []}
        results = {"tool_results": {}, "extracted_features": {}}
        actor_prompt = builder.build_actor_prompt(context, strategy, results)
        assert len(actor_prompt) > 100
        print(f"  [OK] Q1 actor prompt generated ({len(actor_prompt)} chars)")

        # Test Q6 (counterfactual)
        context["target_class"] = "cat"
        builder6 = get_prompt_builder(6, "vision")
        proposer_prompt6 = builder6.build_proposer_prompt(context)
        assert "change" in proposer_prompt6.lower() or "flip" in proposer_prompt6.lower()
        print(f"  [OK] Q6 proposer prompt generated ({len(proposer_prompt6)} chars)")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_masking_utils():
    """Test masking utilities"""
    print("\n" + "=" * 70)
    print("TEST 5: Masking Utilities")
    print("=" * 70)

    try:
        from evaluation.masking_utils import get_masker, MaskingStrategy
        import numpy as np

        # Test vision masker
        masker = get_masker("vision", MaskingStrategy.ZERO)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        region = {"bounding_box": [20, 20, 80, 80]}
        masked = masker.mask(img, region)
        assert masked[50, 50, 0] == 0  # Center should be masked
        assert masked[10, 10, 0] == 255  # Outside region should be unchanged
        print("  [OK] Vision masker (numpy) works")

        # Test text masker
        masker = get_masker("text", MaskingStrategy.MASK_TOKEN)
        text = "Hello world, this is a test."
        region = {"start_index": 6, "end_index": 11}
        masked = masker.mask(text, region)
        assert "[MASK]" in masked
        print("  [OK] Text masker works")

        # Test tabular masker
        masker = get_masker("tabular", MaskingStrategy.ZERO)
        data = {"age": 30, "income": 50000, "score": 85}
        region = {"feature_key": "income"}
        masked = masker.mask(data, region)
        assert masked["income"] == 0
        assert masked["age"] == 30
        print("  [OK] Tabular masker works")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_evaluator_creation():
    """Test evaluator creation (without running evaluation)"""
    print("\n" + "=" * 70)
    print("TEST 6: Evaluator Creation")
    print("=" * 70)

    try:
        from evaluation import get_evaluator

        for q_type in range(1, 11):
            for modality in ["vision", "text", "tabular"]:
                try:
                    evaluator = get_evaluator(q_type, modality)
                    assert evaluator.question_type == q_type
                    assert evaluator.modality == modality
                    assert evaluator.metric_name is not None
                    assert evaluator.metric_formula is not None
                except Exception as e:
                    print(f"  [FAIL] Q{q_type} ({modality}): {e}")
                    return False

        print(f"  [OK] All 30 evaluators created successfully")

        # Check specific metrics
        from evaluation.explanation_faithfulness import Q1Evaluator, Q2Evaluator

        q1 = Q1Evaluator("vision")
        assert q1.metric_formula == "P_original - P_modified"
        print(f"  [OK] Q1 metric: {q1.metric_formula}")

        q2 = Q2Evaluator("vision")
        assert "P_original" in q2.metric_formula
        print(f"  [OK] Q2 metric: {q2.metric_formula}")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of components"""
    print("\n" + "=" * 70)
    print("TEST 7: Integration Test")
    print("=" * 70)

    try:
        from question_templates_new import get_question_template
        from evaluation import get_evaluator

        # Test that template and evaluator work together
        for q_type in [1, 5, 6, 8]:
            template = get_question_template(q_type, "vision")
            evaluator = get_evaluator(q_type, "vision")

            # Verify they are for the same question type
            assert template.q_type == evaluator.question_type
            print(f"  [OK] Q{q_type}: Template and evaluator matched")

        print(f"\n  Integration test passed!")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("TESTING NEW MODULAR ARCHITECTURE")
    print("=" * 70)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Question Templates", test_question_templates()))
    results.append(("Output Schemas", test_output_schemas()))
    results.append(("Prompt Builders", test_prompt_builders()))
    results.append(("Masking Utils", test_masking_utils()))
    results.append(("Evaluator Creation", test_evaluator_creation()))
    results.append(("Integration", test_integration()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed! Architecture is ready for use.")
        return 0
    else:
        print("\n  Some tests failed. Please fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
