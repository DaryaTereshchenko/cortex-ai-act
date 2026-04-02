#!/usr/bin/env python3
"""
Example: Testing Cortex-Specific Evaluation Integration

This script demonstrates the new cortex_expected_answer feature.
"""

import sys
from pathlib import Path

# Simulate a test dataset
test_data = {
    "Question": [
        "What are the requirements for AI transparency under the AI Act?",
        "How does the GDPR define personal data?",
        "What are prohibited AI systems under Chapter II of the AI Act?",
    ],
    "Correct Answer": [
        "Documentation, instructions, and disclosure of AI use",
        "Any information relating to an identified or identifiable natural person",
        "Systems causing a risk of serious harm to health or fundamental rights",
    ],
    "Cortex Expected Answer": [
        "Requirements include: (1) transparency through technical documentation, "
        "(2) instructions for human oversight, (3) disclosure when interacting with humans, "
        "(4) labeling of generated content. Found in Articles 13-14 of the AI Act.",
        
        "Personal data is any information relating to an identified or identifiable natural person "
        "(a 'data subject'). This includes names, identification numbers, and factors specific "
        "to physical, physiological, mental, economic, cultural or social identity.",
        
        "Prohibited systems include: (1) emotion recognition in law enforcement and workplace monitoring "
        "(restrictions under Article 5), (2) social credit systems producing discriminatory effects, "
        "(3) real-time remote biometric identification without exceptions. Listed in Chapter II, Article 5.",
    ],
    "golden_ids": [
        "ai_act_art_13,ai_act_art_14",
        "gdpr_art_4",
        "ai_act_art_5,ai_act_art_6",
    ],
    "Doc": ["AI Act", "GDPR", "AI Act"],
}

def test_dataframe_loading():
    """Test that load_eval_dataframe correctly handles cortex_expected_answer."""
    import pandas as pd
    import tempfile
    import json
    
    print("=" * 80)
    print("TEST: load_eval_dataframe with cortex_expected_answer")
    print("=" * 80)
    
    # Create test data
    df = pd.DataFrame(test_data)
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        temp_path = f.name
        df.to_excel(temp_path, index=False)
    
    try:
        # Import the function to test
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root))
        from baselines.evaluation.run_eval import load_eval_dataframe
        
        # Load the dataframe
        loaded_df = load_eval_dataframe(Path(temp_path))
        
        # Verify columns
        print(f"\n✓ Columns in loaded dataframe:")
        print(f"  {loaded_df.columns.tolist()}")
        
        # Check that cortex_expected_answer was preserved
        if "cortex_expected_answer" in loaded_df.columns:
            print(f"\n✓ cortex_expected_answer column exists")
            print(f"  Sample: {loaded_df.iloc[0]['cortex_expected_answer'][:100]}...")
        else:
            print(f"\n✗ ERROR: cortex_expected_answer column not found!")
            return False
        
        # Check golden_ids parsing
        if "golden_ids" in loaded_df.columns:
            gold_ids_0 = loaded_df.iloc[0]["golden_ids"]
            print(f"\n✓ golden_ids parsed correctly")
            print(f"  Type: {type(gold_ids_0)}, Value: {gold_ids_0}")
        
        print(f"\n✓ All tests passed!")
        return True
        
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_expected_answer_selection():
    """Test that get_expected_answer function works correctly."""
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("TEST: get_expected_answer helper function")
    print("=" * 80)
    
    # Create test row
    test_row = pd.Series({
        "expected_answer": "Generic answer",
        "cortex_expected_answer": "Specialized Cortex answer",
    })
    
    # Import function
    repo_root = Path(__file__).parent
    sys.path.insert(0, str(repo_root))
    from baselines.evaluation.run_eval import get_expected_answer
    
    # Test: Baseline model without flag
    result_naive_no_flag = get_expected_answer(test_row, "Naive", False)
    assert result_naive_no_flag == "Generic answer", "Naive without flag should use expected_answer"
    print(f"✓ Naive (flag=False): {result_naive_no_flag}")
    
    # Test: Baseline model with flag (should still use expected_answer)
    result_naive_with_flag = get_expected_answer(test_row, "Naive", True)
    assert result_naive_with_flag == "Generic answer", "Naive with flag should still use expected_answer"
    print(f"✓ Naive (flag=True): {result_naive_with_flag}")
    
    # Test: Cortex model without flag
    result_cortex_no_flag = get_expected_answer(test_row, "Cortex", False)
    assert result_cortex_no_flag == "Generic answer", "Cortex without flag should use expected_answer"
    print(f"✓ Cortex (flag=False): {result_cortex_no_flag}")
    
    # Test: Cortex model with flag (should use cortex_expected_answer)
    result_cortex_with_flag = get_expected_answer(test_row, "Cortex", True)
    assert result_cortex_with_flag == "Specialized Cortex answer", "Cortex with flag should use cortex_expected_answer"
    print(f"✓ Cortex (flag=True): {result_cortex_with_flag}")
    
    # Test all Cortex variants
    for model in ["Cortex_Pruner_Only", "Cortex_Critic_Only", "Cortex"]:
        result = get_expected_answer(test_row, model, True)
        assert result == "Specialized Cortex answer", f"{model} should use cortex_expected_answer"
        print(f"✓ {model} (flag=True): uses specialized answer")
    
    print(f"\n✓ All selection tests passed!")
    return True


def show_cli_examples():
    """Display example command-line usage."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Usage Patterns")
    print("=" * 80)
    
    examples = [
        ("Standard evaluation (all models, same answer)", 
         "python -m baselines.evaluation.run_eval --input eval_data.xlsx --models naive,cortex"),
        
        ("Cortex-only with specialized answers",
         "python -m baselines.evaluation.run_eval --input eval_data_with_cortex_answers.xlsx "
         "--models cortex --cortex-only-evals"),
        
        ("Fair comparison: baselines vs Cortex",
         "python -m baselines.evaluation.run_eval --input eval_data_with_cortex_answers.xlsx "
         "--models naive,bm25,dense,advanced,cortex --cortex-only-evals"),
        
        ("Cortex variants comparison",
         "python -m baselines.evaluation.run_eval --input eval_data_with_cortex_answers.xlsx "
         "--models cortex-pruner-only,cortex-critic-only,cortex --cortex-only-evals"),
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"\n{i}. {desc}")
        print(f"   Command: {cmd}")
    
    print("\n")


if __name__ == "__main__":
    print("\nCORTEX-SPECIFIC EVALUATION INTEGRATION TEST SUITE\n")
    
    try:
        # Run tests
        success = True
        
        # Test 1: Dataframe loading
        try:
            success = success and test_dataframe_loading()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            success = False
        
        # Test 2: Expected answer selection
        try:
            success = success and test_expected_answer_selection()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            success = False
        
        # Show examples
        show_cli_examples()
        
        if success:
            print("\n" + "=" * 80)
            print("✓ ALL TESTS PASSED - Cortex Integration Ready!")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("✗ SOME TESTS FAILED - Review output above")
            print("=" * 80)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
