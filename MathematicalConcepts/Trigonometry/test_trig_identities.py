"""
Tests for the trigonometry module.

These tests verify that all the trigonometric identities are mathematically correct.
"""

import numpy as np
import sys
import os

# Add the module directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trig_identities import (
    sin_sum, cos_sum, tan_sum,
    sin_diff, cos_diff, tan_diff,
    sin_double, cos_double, tan_double,
    sin_half, cos_half, tan_half,
    get_exact_value, find_angle_decomposition,
    solve_khan_academy_problem
)


def test_sin_sum():
    """Test sine addition identity: sin(α + β) = sin(α)cos(β) + cos(α)sin(β)"""
    test_cases = [
        (45, 30, 75),
        (60, 45, 105),
        (30, 30, 60),
        (0, 45, 45),
        (90, 0, 90),
    ]
    
    for alpha, beta, expected_sum in test_cases:
        result, _ = sin_sum(alpha, beta)
        expected = np.sin(np.radians(expected_sum))
        assert np.isclose(result, expected, rtol=1e-6), \
            f"sin_sum({alpha}, {beta}) = {result}, expected {expected}"
    
    print("✓ sin_sum tests passed")


def test_cos_sum():
    """Test cosine addition identity: cos(α + β) = cos(α)cos(β) - sin(α)sin(β)"""
    test_cases = [
        (45, 30, 75),
        (60, 45, 105),
        (30, 30, 60),
        (0, 45, 45),
        (90, 0, 90),
    ]
    
    for alpha, beta, expected_sum in test_cases:
        result, _ = cos_sum(alpha, beta)
        expected = np.cos(np.radians(expected_sum))
        assert np.isclose(result, expected, rtol=1e-6), \
            f"cos_sum({alpha}, {beta}) = {result}, expected {expected}"
    
    print("✓ cos_sum tests passed")


def test_tan_sum():
    """Test tangent addition identity: tan(α + β) = (tan(α) + tan(β))/(1 - tan(α)tan(β))"""
    test_cases = [
        (45, 30, 75),
        (30, 15, 45),
        (0, 45, 45),
        (20, 25, 45),
    ]
    
    for alpha, beta, expected_sum in test_cases:
        result, _ = tan_sum(alpha, beta)
        expected = np.tan(np.radians(expected_sum))
        assert np.isclose(result, expected, rtol=1e-5), \
            f"tan_sum({alpha}, {beta}) = {result}, expected {expected}"
    
    print("✓ tan_sum tests passed")


def test_sin_diff():
    """Test sine subtraction identity: sin(α - β) = sin(α)cos(β) - cos(α)sin(β)"""
    test_cases = [
        (45, 30, 15),
        (90, 30, 60),
        (60, 45, 15),
        (180, 30, 150),
    ]
    
    for alpha, beta, expected_diff in test_cases:
        result, _ = sin_diff(alpha, beta)
        expected = np.sin(np.radians(expected_diff))
        assert np.isclose(result, expected, rtol=1e-6), \
            f"sin_diff({alpha}, {beta}) = {result}, expected {expected}"
    
    print("✓ sin_diff tests passed")


def test_cos_diff():
    """Test cosine subtraction identity: cos(α - β) = cos(α)cos(β) + sin(α)sin(β)"""
    test_cases = [
        (45, 30, 15),
        (90, 30, 60),
        (60, 45, 15),
        (180, 30, 150),
    ]
    
    for alpha, beta, expected_diff in test_cases:
        result, _ = cos_diff(alpha, beta)
        expected = np.cos(np.radians(expected_diff))
        assert np.isclose(result, expected, rtol=1e-6), \
            f"cos_diff({alpha}, {beta}) = {result}, expected {expected}"
    
    print("✓ cos_diff tests passed")


def test_tan_diff():
    """Test tangent subtraction identity: tan(α - β) = (tan(α) - tan(β))/(1 + tan(α)tan(β))"""
    test_cases = [
        (60, 45, 15),
        (45, 30, 15),
        (90, 45, 45),
    ]
    
    for alpha, beta, expected_diff in test_cases:
        result, _ = tan_diff(alpha, beta)
        expected = np.tan(np.radians(expected_diff))
        assert np.isclose(result, expected, rtol=1e-5), \
            f"tan_diff({alpha}, {beta}) = {result}, expected {expected}"
    
    print("✓ tan_diff tests passed")


def test_sin_double():
    """Test double angle identity: sin(2θ) = 2sin(θ)cos(θ)"""
    test_cases = [30, 45, 60, 15, 22.5, 75]
    
    for theta in test_cases:
        result, _ = sin_double(theta)
        expected = np.sin(np.radians(2 * theta))
        assert np.isclose(result, expected, rtol=1e-6), \
            f"sin_double({theta}) = {result}, expected {expected}"
    
    print("✓ sin_double tests passed")


def test_cos_double():
    """Test double angle identity: cos(2θ) = cos²(θ) - sin²(θ)"""
    test_cases = [30, 45, 60, 15, 22.5, 75]
    
    for theta in test_cases:
        # Test all three forms
        for form in ['default', 'cos_only', 'sin_only']:
            result, _ = cos_double(theta, form=form)
            expected = np.cos(np.radians(2 * theta))
            assert np.isclose(result, expected, rtol=1e-6), \
                f"cos_double({theta}, form={form}) = {result}, expected {expected}"
    
    print("✓ cos_double tests passed")


def test_tan_double():
    """Test double angle identity: tan(2θ) = 2tan(θ)/(1 - tan²(θ))"""
    test_cases = [15, 30, 22.5, 10, 20]
    
    for theta in test_cases:
        result, _ = tan_double(theta)
        expected = np.tan(np.radians(2 * theta))
        assert np.isclose(result, expected, rtol=1e-5), \
            f"tan_double({theta}) = {result}, expected {expected}"
    
    print("✓ tan_double tests passed")


def test_sin_half():
    """Test half angle identity: sin(θ/2) = ±√[(1 - cos(θ))/2]"""
    test_cases = [60, 90, 120, 45, 180]
    
    for theta in test_cases:
        result, _ = sin_half(theta)
        expected = np.sin(np.radians(theta / 2))
        assert np.isclose(result, expected, rtol=1e-6), \
            f"sin_half({theta}) = {result}, expected {expected}"
    
    print("✓ sin_half tests passed")


def test_cos_half():
    """Test half angle identity: cos(θ/2) = ±√[(1 + cos(θ))/2]"""
    test_cases = [60, 90, 120, 45, 180]
    
    for theta in test_cases:
        result, _ = cos_half(theta)
        expected = np.cos(np.radians(theta / 2))
        assert np.isclose(result, expected, rtol=1e-6), \
            f"cos_half({theta}) = {result}, expected {expected}"
    
    print("✓ cos_half tests passed")


def test_tan_half():
    """Test half angle identity: tan(θ/2) = sin(θ)/(1 + cos(θ))"""
    test_cases = [60, 90, 120, 45]
    
    for theta in test_cases:
        result, _ = tan_half(theta)
        expected = np.tan(np.radians(theta / 2))
        assert np.isclose(result, expected, rtol=1e-5), \
            f"tan_half({theta}) = {result}, expected {expected}"
    
    print("✓ tan_half tests passed")


def test_pythagorean_identity():
    """Test that sin²(θ) + cos²(θ) = 1 for all angles"""
    test_angles = np.linspace(0, 360, 37)
    
    for theta in test_angles:
        theta_rad = np.radians(theta)
        result = np.sin(theta_rad)**2 + np.cos(theta_rad)**2
        assert np.isclose(result, 1.0, rtol=1e-10), \
            f"sin²({theta}) + cos²({theta}) = {result}, expected 1.0"
    
    print("✓ pythagorean_identity tests passed")


def test_angle_decomposition():
    """Test that angle decomposition finds valid decompositions"""
    test_cases = {
        75: [(45, 30, '+')],  # 75 = 45 + 30
        15: [(45, 30, '-')],  # 15 = 45 - 30
        105: [(60, 45, '+')],  # 105 = 60 + 45
        165: [(120, 45, '+')],  # 165 = 120 + 45
    }
    
    for angle, expected_decomps in test_cases.items():
        decompositions = find_angle_decomposition(angle)
        assert len(decompositions) > 0, f"No decomposition found for {angle}°"
        
        # Check that at least one expected decomposition is found
        found = False
        for exp in expected_decomps:
            if exp in decompositions:
                found = True
                break
        
        # Verify the decomposition is mathematically correct
        for alpha, beta, op in decompositions:
            if op == '+':
                assert alpha + beta == angle, f"{alpha} + {beta} != {angle}"
            else:
                assert alpha - beta == angle, f"{alpha} - {beta} != {angle}"
    
    print("✓ angle_decomposition tests passed")


def test_special_angle_values():
    """Test that special angle exact values match computed values"""
    special_angles = {
        0: (0, 1),
        30: (0.5, np.sqrt(3)/2),
        45: (np.sqrt(2)/2, np.sqrt(2)/2),
        60: (np.sqrt(3)/2, 0.5),
        90: (1, 0),
    }
    
    for angle, (sin_val, cos_val) in special_angles.items():
        theta_rad = np.radians(angle)
        
        computed_sin = np.sin(theta_rad)
        computed_cos = np.cos(theta_rad)
        
        assert np.isclose(computed_sin, sin_val, rtol=1e-10), \
            f"sin({angle}) = {computed_sin}, expected {sin_val}"
        assert np.isclose(computed_cos, cos_val, rtol=1e-10), \
            f"cos({angle}) = {computed_cos}, expected {cos_val}"
    
    print("✓ special_angle_values tests passed")


def test_radians_mode():
    """Test that radians mode works correctly"""
    # Test with radians
    result_rad, _ = sin_sum(np.pi/4, np.pi/6, degrees=False)
    result_deg, _ = sin_sum(45, 30, degrees=True)
    
    assert np.isclose(result_rad, result_deg, rtol=1e-10), \
        f"Radians mode: {result_rad}, Degrees mode: {result_deg}"
    
    print("✓ radians_mode tests passed")


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("RUNNING TRIGONOMETRY MODULE TESTS")
    print("="*60 + "\n")
    
    tests = [
        test_sin_sum,
        test_cos_sum,
        test_tan_sum,
        test_sin_diff,
        test_cos_diff,
        test_tan_diff,
        test_sin_double,
        test_cos_double,
        test_tan_double,
        test_sin_half,
        test_cos_half,
        test_tan_half,
        test_pythagorean_identity,
        test_angle_decomposition,
        test_special_angle_values,
        test_radians_mode,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    # Suppress the print statements from the functions during testing
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        success = run_all_tests()
    
    # Now print the test results
    run_all_tests()
    
    exit(0 if success else 1)
