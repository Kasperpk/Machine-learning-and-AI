"""
Trigonometry Module: Angle Addition Identities and Beyond

This module helps you understand and solve trigonometric problems like those on Khan Academy.
It provides:
- Angle addition/subtraction identities
- Double angle and half angle identities  
- Step-by-step intuition and explanations
- Visualizations to build geometric understanding

Key Insight: All these identities come from the unit circle and how we measure
distances and angles. The angle addition formula, in particular, can be derived
by thinking about rotating points on the unit circle.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from fractions import Fraction


# =============================================================================
# INTUITION BUILDERS
# =============================================================================

def explain_angle_addition():
    """
    Explains WHY the angle addition formulas work.
    
    The key intuition:
    When we have sin(α + β), we're asking: "If I rotate by angle α, then by angle β,
    what is the y-coordinate of where I end up on the unit circle?"
    
    The formulas sin(α + β) = sin(α)cos(β) + cos(α)sin(β) comes from:
    1. Representing rotation as complex number multiplication: e^(iα) × e^(iβ) = e^(i(α+β))
    2. Or geometrically: projecting the rotated point onto the axes
    """
    explanation = """
    ═══════════════════════════════════════════════════════════════════════════
    UNDERSTANDING ANGLE ADDITION IDENTITIES
    ═══════════════════════════════════════════════════════════════════════════
    
    THE BIG PICTURE:
    ───────────────
    Imagine standing at the origin of a coordinate system, pointing at (1, 0).
    
    • sin(θ) tells you the y-coordinate when you rotate by angle θ
    • cos(θ) tells you the x-coordinate when you rotate by angle θ
    
    THE QUESTION:
    ─────────────
    If I rotate first by α, then by β, where do I end up?
    
    The answer involves both angles working together:
    
    sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
                 ↑             ↑
                 Part from α   Part from β
    
    cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
                 ↑             ↑
                 Part from α   Part from β
    
    GEOMETRIC INTUITION:
    ────────────────────
    Think of it as two "contributions":
    
    For sin(α + β) (the vertical position):
    • sin(α)cos(β): The vertical part of α, scaled by how much β "preserves" it
    • cos(α)sin(β): The horizontal part of α that gets "converted" to vertical by β
    
    WHY THE MINUS SIGN IN COSINE?
    ─────────────────────────────
    In cos(α + β) = cos(α)cos(β) - sin(α)sin(β):
    • When both angles push you away from the x-axis (both sines positive),
      you end up further left (more negative x), hence the minus sign
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    print(explanation)
    return explanation


def explain_unit_circle():
    """
    Explains the fundamental connection between the unit circle and trig functions.
    """
    explanation = """
    ═══════════════════════════════════════════════════════════════════════════
    THE UNIT CIRCLE: THE FOUNDATION OF TRIGONOMETRY
    ═══════════════════════════════════════════════════════════════════════════
    
    The unit circle is a circle with radius 1, centered at the origin.
    
    For any angle θ measured from the positive x-axis:
    
    • The point on the circle is: (cos(θ), sin(θ))
    • x-coordinate = cos(θ) (horizontal distance)
    • y-coordinate = sin(θ) (vertical distance)
    
    KEY SPECIAL ANGLES:
    ───────────────────
    θ = 0°    → (1, 0)        → sin = 0,        cos = 1
    θ = 30°   → (√3/2, 1/2)   → sin = 1/2,      cos = √3/2
    θ = 45°   → (√2/2, √2/2)  → sin = √2/2,     cos = √2/2
    θ = 60°   → (1/2, √3/2)   → sin = √3/2,     cos = 1/2
    θ = 90°   → (0, 1)        → sin = 1,        cos = 0
    
    MEMORY TRICK:
    ─────────────
    For 0°, 30°, 45°, 60°, 90°:
    sin values: √0/2, √1/2, √2/2, √3/2, √4/2 = 0, 1/2, √2/2, √3/2, 1
    cos values: go in reverse!
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    print(explanation)
    return explanation


# =============================================================================
# ANGLE ADDITION IDENTITIES
# =============================================================================

def sin_sum(alpha: float, beta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate sin(α + β) using the angle addition identity.
    
    sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
    
    Parameters:
    -----------
    alpha : float
        First angle
    beta : float  
        Second angle
    degrees : bool
        If True, angles are in degrees; if False, radians
        
    Returns:
    --------
    Tuple[float, str]
        The result and a step-by-step explanation
    """
    if degrees:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
    else:
        alpha_rad = alpha
        beta_rad = beta
    
    # Components
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    sin_b = np.sin(beta_rad)
    cos_b = np.cos(beta_rad)
    
    # The identity
    result = sin_a * cos_b + cos_a * sin_b
    
    # Verification using direct computation
    direct = np.sin(alpha_rad + beta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: sin({alpha}{unit} + {beta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the identity: sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
    
    Step 1: Find the components
       sin({alpha}{unit}) = {sin_a:.6f}
       cos({alpha}{unit}) = {cos_a:.6f}
       sin({beta}{unit}) = {sin_b:.6f}
       cos({beta}{unit}) = {cos_b:.6f}
    
    Step 2: Apply the formula
       sin({alpha}{unit})·cos({beta}{unit}) + cos({alpha}{unit})·sin({beta}{unit})
       = ({sin_a:.6f})·({cos_b:.6f}) + ({cos_a:.6f})·({sin_b:.6f})
       = {sin_a * cos_b:.6f} + {cos_a * sin_b:.6f}
       = {result:.6f}
    
    ✓ Verification: sin({alpha + beta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def cos_sum(alpha: float, beta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate cos(α + β) using the angle addition identity.
    
    cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
    
    Parameters:
    -----------
    alpha : float
        First angle
    beta : float  
        Second angle
    degrees : bool
        If True, angles are in degrees; if False, radians
        
    Returns:
    --------
    Tuple[float, str]
        The result and a step-by-step explanation
    """
    if degrees:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
    else:
        alpha_rad = alpha
        beta_rad = beta
    
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    sin_b = np.sin(beta_rad)
    cos_b = np.cos(beta_rad)
    
    result = cos_a * cos_b - sin_a * sin_b
    direct = np.cos(alpha_rad + beta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: cos({alpha}{unit} + {beta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the identity: cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
    
    Step 1: Find the components
       sin({alpha}{unit}) = {sin_a:.6f}
       cos({alpha}{unit}) = {cos_a:.6f}
       sin({beta}{unit}) = {sin_b:.6f}
       cos({beta}{unit}) = {cos_b:.6f}
    
    Step 2: Apply the formula
       cos({alpha}{unit})·cos({beta}{unit}) - sin({alpha}{unit})·sin({beta}{unit})
       = ({cos_a:.6f})·({cos_b:.6f}) - ({sin_a:.6f})·({sin_b:.6f})
       = {cos_a * cos_b:.6f} - {sin_a * sin_b:.6f}
       = {result:.6f}
    
    ✓ Verification: cos({alpha + beta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def tan_sum(alpha: float, beta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate tan(α + β) using the angle addition identity.
    
    tan(α + β) = (tan(α) + tan(β)) / (1 - tan(α)tan(β))
    
    Parameters:
    -----------
    alpha : float
        First angle
    beta : float  
        Second angle
    degrees : bool
        If True, angles are in degrees; if False, radians
        
    Returns:
    --------
    Tuple[float, str]
        The result and a step-by-step explanation
    """
    if degrees:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
    else:
        alpha_rad = alpha
        beta_rad = beta
    
    tan_a = np.tan(alpha_rad)
    tan_b = np.tan(beta_rad)
    
    denominator = 1 - tan_a * tan_b
    if np.abs(denominator) < 1e-10:
        return float('inf'), "The result is undefined (denominator is zero)"
    
    result = (tan_a + tan_b) / denominator
    direct = np.tan(alpha_rad + beta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: tan({alpha}{unit} + {beta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the identity: tan(α + β) = (tan(α) + tan(β)) / (1 - tan(α)tan(β))
    
    Step 1: Find the tangent values
       tan({alpha}{unit}) = {tan_a:.6f}
       tan({beta}{unit}) = {tan_b:.6f}
    
    Step 2: Apply the formula
       Numerator: tan({alpha}{unit}) + tan({beta}{unit})
                = {tan_a:.6f} + {tan_b:.6f}
                = {tan_a + tan_b:.6f}
       
       Denominator: 1 - tan({alpha}{unit})·tan({beta}{unit})
                  = 1 - ({tan_a:.6f})·({tan_b:.6f})
                  = 1 - {tan_a * tan_b:.6f}
                  = {denominator:.6f}
       
       Result: {tan_a + tan_b:.6f} / {denominator:.6f} = {result:.6f}
    
    ✓ Verification: tan({alpha + beta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


# =============================================================================
# ANGLE SUBTRACTION IDENTITIES  
# =============================================================================

def sin_diff(alpha: float, beta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate sin(α - β) using the angle subtraction identity.
    
    sin(α - β) = sin(α)cos(β) - cos(α)sin(β)
    """
    if degrees:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
    else:
        alpha_rad = alpha
        beta_rad = beta
    
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    sin_b = np.sin(beta_rad)
    cos_b = np.cos(beta_rad)
    
    result = sin_a * cos_b - cos_a * sin_b
    direct = np.sin(alpha_rad - beta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: sin({alpha}{unit} - {beta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the identity: sin(α - β) = sin(α)cos(β) - cos(α)sin(β)
    
    Note: This is similar to sin(α + β), but with a MINUS instead of plus.
    Think of it as "undoing" part of the rotation.
    
    Step 1: Find the components
       sin({alpha}{unit}) = {sin_a:.6f}
       cos({alpha}{unit}) = {cos_a:.6f}
       sin({beta}{unit}) = {sin_b:.6f}
       cos({beta}{unit}) = {cos_b:.6f}
    
    Step 2: Apply the formula
       = ({sin_a:.6f})·({cos_b:.6f}) - ({cos_a:.6f})·({sin_b:.6f})
       = {sin_a * cos_b:.6f} - {cos_a * sin_b:.6f}
       = {result:.6f}
    
    ✓ Verification: sin({alpha - beta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def cos_diff(alpha: float, beta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate cos(α - β) using the angle subtraction identity.
    
    cos(α - β) = cos(α)cos(β) + sin(α)sin(β)
    """
    if degrees:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
    else:
        alpha_rad = alpha
        beta_rad = beta
    
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    sin_b = np.sin(beta_rad)
    cos_b = np.cos(beta_rad)
    
    result = cos_a * cos_b + sin_a * sin_b
    direct = np.cos(alpha_rad - beta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: cos({alpha}{unit} - {beta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the identity: cos(α - β) = cos(α)cos(β) + sin(α)sin(β)
    
    Note: This is similar to cos(α + β), but with a PLUS instead of minus.
    When we subtract an angle, the cross terms reinforce rather than cancel.
    
    Step 1: Find the components
       sin({alpha}{unit}) = {sin_a:.6f}
       cos({alpha}{unit}) = {cos_a:.6f}
       sin({beta}{unit}) = {sin_b:.6f}
       cos({beta}{unit}) = {cos_b:.6f}
    
    Step 2: Apply the formula
       = ({cos_a:.6f})·({cos_b:.6f}) + ({sin_a:.6f})·({sin_b:.6f})
       = {cos_a * cos_b:.6f} + {sin_a * sin_b:.6f}
       = {result:.6f}
    
    ✓ Verification: cos({alpha - beta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def tan_diff(alpha: float, beta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate tan(α - β) using the angle subtraction identity.
    
    tan(α - β) = (tan(α) - tan(β)) / (1 + tan(α)tan(β))
    """
    if degrees:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
    else:
        alpha_rad = alpha
        beta_rad = beta
    
    tan_a = np.tan(alpha_rad)
    tan_b = np.tan(beta_rad)
    
    denominator = 1 + tan_a * tan_b
    if np.abs(denominator) < 1e-10:
        return float('inf'), "The result is undefined (denominator is zero)"
    
    result = (tan_a - tan_b) / denominator
    direct = np.tan(alpha_rad - beta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: tan({alpha}{unit} - {beta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the identity: tan(α - β) = (tan(α) - tan(β)) / (1 + tan(α)tan(β))
    
    Step 1: Find the tangent values
       tan({alpha}{unit}) = {tan_a:.6f}
       tan({beta}{unit}) = {tan_b:.6f}
    
    Step 2: Apply the formula
       Numerator: {tan_a:.6f} - {tan_b:.6f} = {tan_a - tan_b:.6f}
       Denominator: 1 + ({tan_a:.6f})·({tan_b:.6f}) = {denominator:.6f}
       Result: {result:.6f}
    
    ✓ Verification: tan({alpha - beta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


# =============================================================================
# DOUBLE ANGLE IDENTITIES
# =============================================================================

def sin_double(theta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate sin(2θ) using the double angle identity.
    
    sin(2θ) = 2·sin(θ)·cos(θ)
    
    This comes directly from sin(θ + θ) = sin(θ)cos(θ) + cos(θ)sin(θ) = 2sin(θ)cos(θ)
    """
    if degrees:
        theta_rad = np.radians(theta)
    else:
        theta_rad = theta
    
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    
    result = 2 * sin_t * cos_t
    direct = np.sin(2 * theta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: sin(2 × {theta}{unit}) = sin({2*theta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the double angle identity: sin(2θ) = 2·sin(θ)·cos(θ)
    
    INTUITION: This comes from sin(θ + θ):
       sin(θ + θ) = sin(θ)cos(θ) + cos(θ)sin(θ) = 2sin(θ)cos(θ)
    
    Step 1: Find sin(θ) and cos(θ)
       sin({theta}{unit}) = {sin_t:.6f}
       cos({theta}{unit}) = {cos_t:.6f}
    
    Step 2: Apply the formula
       2 × ({sin_t:.6f}) × ({cos_t:.6f})
       = 2 × {sin_t * cos_t:.6f}
       = {result:.6f}
    
    ✓ Verification: sin({2*theta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def cos_double(theta: float, degrees: bool = True, form: str = 'default') -> Tuple[float, str]:
    """
    Calculate cos(2θ) using the double angle identity.
    
    There are THREE equivalent forms:
    1. cos(2θ) = cos²(θ) - sin²(θ)     [default]
    2. cos(2θ) = 2cos²(θ) - 1          [cos_only]
    3. cos(2θ) = 1 - 2sin²(θ)          [sin_only]
    
    Parameters:
    -----------
    form : str
        'default' - uses cos²θ - sin²θ
        'cos_only' - uses 2cos²θ - 1
        'sin_only' - uses 1 - 2sin²θ
    """
    if degrees:
        theta_rad = np.radians(theta)
    else:
        theta_rad = theta
    
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    
    direct = np.cos(2 * theta_rad)
    unit = "°" if degrees else " rad"
    
    if form == 'cos_only':
        result = 2 * cos_t**2 - 1
        formula_str = "2cos²(θ) - 1"
        calc_str = f"2 × ({cos_t:.6f})² - 1 = 2 × {cos_t**2:.6f} - 1 = {result:.6f}"
    elif form == 'sin_only':
        result = 1 - 2 * sin_t**2
        formula_str = "1 - 2sin²(θ)"
        calc_str = f"1 - 2 × ({sin_t:.6f})² = 1 - 2 × {sin_t**2:.6f} = {result:.6f}"
    else:
        result = cos_t**2 - sin_t**2
        formula_str = "cos²(θ) - sin²(θ)"
        calc_str = f"({cos_t:.6f})² - ({sin_t:.6f})² = {cos_t**2:.6f} - {sin_t**2:.6f} = {result:.6f}"
    
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: cos(2 × {theta}{unit}) = cos({2*theta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the double angle identity: cos(2θ) = {formula_str}
    
    THREE EQUIVALENT FORMS:
    • cos²(θ) - sin²(θ)   ← difference of squares
    • 2cos²(θ) - 1        ← useful when you only know cosine
    • 1 - 2sin²(θ)        ← useful when you only know sine
    
    Step 1: Find sin(θ) and cos(θ)
       sin({theta}{unit}) = {sin_t:.6f}
       cos({theta}{unit}) = {cos_t:.6f}
    
    Step 2: Apply the formula ({formula_str})
       {calc_str}
    
    ✓ Verification: cos({2*theta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def tan_double(theta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate tan(2θ) using the double angle identity.
    
    tan(2θ) = 2tan(θ) / (1 - tan²(θ))
    """
    if degrees:
        theta_rad = np.radians(theta)
    else:
        theta_rad = theta
    
    tan_t = np.tan(theta_rad)
    
    denominator = 1 - tan_t**2
    if np.abs(denominator) < 1e-10:
        return float('inf'), "The result is undefined (denominator is zero)"
    
    result = 2 * tan_t / denominator
    direct = np.tan(2 * theta_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: tan(2 × {theta}{unit}) = tan({2*theta}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the double angle identity: tan(2θ) = 2tan(θ) / (1 - tan²(θ))
    
    Step 1: Find tan(θ)
       tan({theta}{unit}) = {tan_t:.6f}
    
    Step 2: Apply the formula
       Numerator: 2 × {tan_t:.6f} = {2*tan_t:.6f}
       Denominator: 1 - ({tan_t:.6f})² = 1 - {tan_t**2:.6f} = {denominator:.6f}
       Result: {2*tan_t:.6f} / {denominator:.6f} = {result:.6f}
    
    ✓ Verification: tan({2*theta}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


# =============================================================================
# HALF ANGLE IDENTITIES
# =============================================================================

def sin_half(theta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate sin(θ/2) using the half angle identity.
    
    sin(θ/2) = ±√[(1 - cos(θ))/2]
    
    The sign depends on which quadrant θ/2 is in.
    """
    if degrees:
        theta_rad = np.radians(theta)
        half_angle = theta / 2
    else:
        theta_rad = theta
        half_angle = theta / 2
    
    cos_t = np.cos(theta_rad)
    
    # Determine sign based on quadrant of θ/2
    if degrees:
        half_rad = np.radians(half_angle)
    else:
        half_rad = half_angle
    
    # sin is positive in Q1 and Q2
    sign = 1 if np.sin(half_rad) >= 0 else -1
    
    result = sign * np.sqrt((1 - cos_t) / 2)
    direct = np.sin(half_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: sin({theta}{unit}/2) = sin({half_angle}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the half angle identity: sin(θ/2) = ±√[(1 - cos(θ))/2]
    
    INTUITION: This comes from solving cos(2θ) = 1 - 2sin²(θ) for sin(θ)
    
    Step 1: Find cos(θ)
       cos({theta}{unit}) = {cos_t:.6f}
    
    Step 2: Calculate (1 - cos(θ))/2
       (1 - {cos_t:.6f})/2 = {(1-cos_t)/2:.6f}
    
    Step 3: Take square root with correct sign
       The angle {half_angle}{unit} is in a quadrant where sin is {'positive' if sign > 0 else 'negative'}
       sin({half_angle}{unit}) = {'+' if sign > 0 else '-'}√{(1-cos_t)/2:.6f} = {result:.6f}
    
    ✓ Verification: sin({half_angle}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def cos_half(theta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate cos(θ/2) using the half angle identity.
    
    cos(θ/2) = ±√[(1 + cos(θ))/2]
    
    The sign depends on which quadrant θ/2 is in.
    """
    if degrees:
        theta_rad = np.radians(theta)
        half_angle = theta / 2
    else:
        theta_rad = theta
        half_angle = theta / 2
    
    cos_t = np.cos(theta_rad)
    
    if degrees:
        half_rad = np.radians(half_angle)
    else:
        half_rad = half_angle
    
    # cos is positive in Q1 and Q4
    sign = 1 if np.cos(half_rad) >= 0 else -1
    
    result = sign * np.sqrt((1 + cos_t) / 2)
    direct = np.cos(half_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: cos({theta}{unit}/2) = cos({half_angle}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the half angle identity: cos(θ/2) = ±√[(1 + cos(θ))/2]
    
    INTUITION: This comes from solving cos(2θ) = 2cos²(θ) - 1 for cos(θ)
    
    Step 1: Find cos(θ)
       cos({theta}{unit}) = {cos_t:.6f}
    
    Step 2: Calculate (1 + cos(θ))/2
       (1 + {cos_t:.6f})/2 = {(1+cos_t)/2:.6f}
    
    Step 3: Take square root with correct sign
       The angle {half_angle}{unit} is in a quadrant where cos is {'positive' if sign > 0 else 'negative'}
       cos({half_angle}{unit}) = {'+' if sign > 0 else '-'}√{(1+cos_t)/2:.6f} = {result:.6f}
    
    ✓ Verification: cos({half_angle}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


def tan_half(theta: float, degrees: bool = True) -> Tuple[float, str]:
    """
    Calculate tan(θ/2) using the half angle identity.
    
    There are multiple forms:
    tan(θ/2) = sin(θ)/(1 + cos(θ)) = (1 - cos(θ))/sin(θ)
    """
    if degrees:
        theta_rad = np.radians(theta)
        half_angle = theta / 2
    else:
        theta_rad = theta
        half_angle = theta / 2
    
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    
    # Use the form that avoids division by zero
    if np.abs(1 + cos_t) > 1e-10:
        result = sin_t / (1 + cos_t)
        formula_used = "sin(θ)/(1 + cos(θ))"
    else:
        result = (1 - cos_t) / sin_t
        formula_used = "(1 - cos(θ))/sin(θ)"
    
    if degrees:
        half_rad = np.radians(half_angle)
    else:
        half_rad = half_angle
    
    direct = np.tan(half_rad)
    
    unit = "°" if degrees else " rad"
    explanation = f"""
    ─────────────────────────────────────────────────────────────
    SOLVING: tan({theta}{unit}/2) = tan({half_angle}{unit})
    ─────────────────────────────────────────────────────────────
    
    Using the half angle identity: tan(θ/2) = {formula_used}
    
    Alternative forms:
    • sin(θ)/(1 + cos(θ))
    • (1 - cos(θ))/sin(θ)
    • ±√[(1 - cos(θ))/(1 + cos(θ))]
    
    Step 1: Find sin(θ) and cos(θ)
       sin({theta}{unit}) = {sin_t:.6f}
       cos({theta}{unit}) = {cos_t:.6f}
    
    Step 2: Apply the formula
       {formula_used} = {result:.6f}
    
    ✓ Verification: tan({half_angle}{unit}) = {direct:.6f}
    ─────────────────────────────────────────────────────────────
    """
    
    print(explanation)
    return result, explanation


# =============================================================================
# EXACT VALUES FOR SPECIAL ANGLES
# =============================================================================

# Special angle values as symbolic strings
SPECIAL_ANGLES = {
    0: {'sin': '0', 'cos': '1', 'tan': '0'},
    30: {'sin': '1/2', 'cos': '√3/2', 'tan': '√3/3'},
    45: {'sin': '√2/2', 'cos': '√2/2', 'tan': '1'},
    60: {'sin': '√3/2', 'cos': '1/2', 'tan': '√3'},
    90: {'sin': '1', 'cos': '0', 'tan': 'undefined'},
    120: {'sin': '√3/2', 'cos': '-1/2', 'tan': '-√3'},
    135: {'sin': '√2/2', 'cos': '-√2/2', 'tan': '-1'},
    150: {'sin': '1/2', 'cos': '-√3/2', 'tan': '-√3/3'},
    180: {'sin': '0', 'cos': '-1', 'tan': '0'},
    210: {'sin': '-1/2', 'cos': '-√3/2', 'tan': '√3/3'},
    225: {'sin': '-√2/2', 'cos': '-√2/2', 'tan': '1'},
    240: {'sin': '-√3/2', 'cos': '-1/2', 'tan': '√3'},
    270: {'sin': '-1', 'cos': '0', 'tan': 'undefined'},
    300: {'sin': '-√3/2', 'cos': '1/2', 'tan': '-√3'},
    315: {'sin': '-√2/2', 'cos': '√2/2', 'tan': '-1'},
    330: {'sin': '-1/2', 'cos': '√3/2', 'tan': '-√3/3'},
    360: {'sin': '0', 'cos': '1', 'tan': '0'},
}


def get_exact_value(angle: float, func: str = 'sin') -> str:
    """
    Get the exact symbolic value for a special angle.
    
    Parameters:
    -----------
    angle : float
        Angle in degrees (0, 30, 45, 60, 90, etc.)
    func : str
        'sin', 'cos', or 'tan'
        
    Returns:
    --------
    str
        The exact symbolic value or 'No exact form' if not a special angle
    """
    angle_normalized = angle % 360
    if angle_normalized in SPECIAL_ANGLES:
        return SPECIAL_ANGLES[angle_normalized][func]
    return f"No exact form (≈ {np.sin(np.radians(angle)) if func == 'sin' else np.cos(np.radians(angle)) if func == 'cos' else np.tan(np.radians(angle)):.6f})"


def print_special_angles_table():
    """Print a table of all special angle values."""
    table = """
    ═══════════════════════════════════════════════════════════════════════════
    SPECIAL ANGLES TABLE
    ═══════════════════════════════════════════════════════════════════════════
    
    │ Angle │   sin(θ)   │   cos(θ)   │   tan(θ)    │
    │───────│────────────│────────────│─────────────│
    │  0°   │     0      │     1      │      0      │
    │ 30°   │    1/2     │    √3/2    │    √3/3     │
    │ 45°   │   √2/2     │    √2/2    │      1      │
    │ 60°   │   √3/2     │    1/2     │     √3      │
    │ 90°   │     1      │     0      │  undefined  │
    │ 120°  │   √3/2     │   -1/2     │    -√3      │
    │ 135°  │   √2/2     │   -√2/2    │     -1      │
    │ 150°  │    1/2     │   -√3/2    │   -√3/3     │
    │ 180°  │     0      │    -1      │      0      │
    
    MEMORY TRICKS:
    ─────────────
    • sin values for 0°, 30°, 45°, 60°, 90°: √0/2, √1/2, √2/2, √3/2, √4/2
    • cos values are sin values in reverse order
    • tan = sin/cos
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    print(table)
    return table


# =============================================================================
# KHAN ACADEMY STYLE PROBLEM SOLVER
# =============================================================================

def solve_khan_academy_problem(expression: str, verbose: bool = True) -> float:
    """
    Solve a Khan Academy style angle addition problem.
    
    Examples:
    ---------
    >>> solve_khan_academy_problem("sin(75)")   # sin(45° + 30°)
    >>> solve_khan_academy_problem("cos(15)")   # cos(45° - 30°)
    >>> solve_khan_academy_problem("tan(105)")  # tan(60° + 45°)
    
    Parameters:
    -----------
    expression : str
        A trigonometric expression like "sin(75)", "cos(15)", "tan(105)"
    verbose : bool
        Whether to print the step-by-step solution
        
    Returns:
    --------
    float
        The computed value
    """
    import re
    
    # Parse the expression
    match = re.match(r'(sin|cos|tan)\((\-?\d+\.?\d*)\)', expression.replace(' ', ''))
    if not match:
        print(f"Could not parse expression: {expression}")
        print("Expected format: sin(angle), cos(angle), or tan(angle)")
        return float('nan')
    
    func, angle_str = match.groups()
    angle = float(angle_str)
    
    # Find a decomposition using special angles
    decompositions = find_angle_decomposition(angle)
    
    if verbose:
        print(f"\n{'='*65}")
        print(f"SOLVING: {expression}°")
        print(f"{'='*65}")
    
    if decompositions:
        alpha, beta, operation = decompositions[0]
        
        if verbose:
            print(f"\nSTRATEGY: Express {angle}° as {alpha}° {operation} {beta}°")
            print(f"\nUsing the angle {'addition' if operation == '+' else 'subtraction'} identity:\n")
        
        if operation == '+':
            if func == 'sin':
                result, _ = sin_sum(alpha, beta, degrees=True)
            elif func == 'cos':
                result, _ = cos_sum(alpha, beta, degrees=True)
            else:
                result, _ = tan_sum(alpha, beta, degrees=True)
        else:
            if func == 'sin':
                result, _ = sin_diff(alpha, beta, degrees=True)
            elif func == 'cos':
                result, _ = cos_diff(alpha, beta, degrees=True)
            else:
                result, _ = tan_diff(alpha, beta, degrees=True)
        
        return result
    else:
        # No decomposition found, compute directly
        if verbose:
            print(f"\nNo special angle decomposition found. Computing directly:")
        
        angle_rad = np.radians(angle)
        if func == 'sin':
            result = np.sin(angle_rad)
        elif func == 'cos':
            result = np.cos(angle_rad)
        else:
            result = np.tan(angle_rad)
        
        if verbose:
            print(f"   {func}({angle}°) = {result:.6f}")
        
        return result


def find_angle_decomposition(angle: float) -> list:
    """
    Find ways to express an angle as a sum or difference of special angles.
    
    Returns a list of tuples: [(alpha, beta, operation), ...]
    where operation is '+' or '-'
    """
    special = [0, 30, 45, 60, 90, 120, 135, 150, 180]
    decompositions = []
    
    for a in special:
        for b in special:
            if a + b == angle:
                decompositions.append((a, b, '+'))
            if a - b == angle and a != b:
                decompositions.append((a, b, '-'))
            if b - a == angle and a != b:
                decompositions.append((b, a, '-'))
    
    # Sort to prefer simpler decompositions
    decompositions.sort(key=lambda x: abs(x[0]) + abs(x[1]))
    
    return decompositions


# =============================================================================
# PRACTICE PROBLEMS GENERATOR
# =============================================================================

def generate_practice_problems(n: int = 5, difficulty: str = 'medium', seed: int = None) -> list:
    """
    Generate Khan Academy style practice problems.
    
    Parameters:
    -----------
    n : int
        Number of problems to generate
    difficulty : str
        'easy' - uses only sin and cos with simple angle sums
        'medium' - includes tan and angle differences  
        'hard' - includes all identities and non-obvious decompositions
    seed : int, optional
        Random seed for reproducible results. If None, uses random selection.
    
    Returns:
    --------
    list
        List of problem dictionaries with 'expression' and 'answer'
    """
    problems = []
    
    if difficulty == 'easy':
        angles = [75, 105, 15, 165]  # 45+30, 60+45, 45-30, 180-15
        funcs = ['sin', 'cos']
    elif difficulty == 'medium':
        angles = [75, 105, 15, 165, 195, 255, 285, 345]
        funcs = ['sin', 'cos', 'tan']
    else:  # hard
        angles = [75, 105, 15, 165, 195, 255, 285, 345, 7.5, 22.5, 37.5, 52.5, 67.5]
        funcs = ['sin', 'cos', 'tan']
    
    rng = np.random.default_rng(seed)
    selected_angles = rng.choice(angles, min(n, len(angles)), replace=False)
    
    for angle in selected_angles:
        func = rng.choice(funcs)
        expression = f"{func}({int(angle) if angle == int(angle) else angle})"
        
        # Calculate answer
        angle_rad = np.radians(angle)
        if func == 'sin':
            answer = np.sin(angle_rad)
        elif func == 'cos':
            answer = np.cos(angle_rad)
        else:
            answer = np.tan(angle_rad)
        
        problems.append({
            'expression': expression,
            'answer': answer,
            'decomposition': find_angle_decomposition(angle)
        })
    
    return problems


def run_practice_session(n: int = 5, difficulty: str = 'medium'):
    """
    Run an interactive practice session.
    """
    problems = generate_practice_problems(n, difficulty)
    
    print("\n" + "="*65)
    print("TRIGONOMETRY PRACTICE SESSION")
    print("="*65)
    print(f"\nDifficulty: {difficulty.upper()}")
    print(f"Problems: {n}")
    print("\nFor each problem, try to find the exact value using angle identities.")
    print("Then press Enter to see the solution.\n")
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'─'*65}")
        print(f"PROBLEM {i}: Find {problem['expression']}°")
        
        if problem['decomposition']:
            alpha, beta, op = problem['decomposition'][0]
            print(f"\nHINT: Can you express this as a sum or difference of special angles?")
            print(f"      Think about {alpha}° and {beta}°...")
        
        print(f"\n[Solution: {problem['answer']:.6f}]")
        
        # Show full solution
        solve_khan_academy_problem(problem['expression'])
    
    print("\n" + "="*65)
    print("Practice session complete!")
    print("="*65)


if __name__ == "__main__":
    # Demo the module
    print("\n" + "="*70)
    print("TRIGONOMETRY IDENTITIES MODULE DEMO")
    print("="*70)
    
    # Show intuition
    explain_angle_addition()
    
    # Example problems
    print("\n" + "="*70)
    print("EXAMPLE: Solving sin(75°)")
    print("="*70)
    solve_khan_academy_problem("sin(75)")
    
    print("\n" + "="*70)
    print("EXAMPLE: Solving cos(15°)")
    print("="*70)
    solve_khan_academy_problem("cos(15)")
    
    # Print special angles
    print_special_angles_table()
