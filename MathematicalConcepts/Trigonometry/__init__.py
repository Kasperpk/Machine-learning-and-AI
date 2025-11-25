"""
Trigonometry Module for Khan Academy Style Problem Solving

This module provides tools for understanding and solving trigonometric problems:

- Angle addition/subtraction identities
- Double angle and half angle identities
- Step-by-step explanations with intuition
- Visualizations for geometric understanding

Usage:
    from MathematicalConcepts.Trigonometry import (
        solve_khan_academy_problem,
        sin_sum, cos_sum, tan_sum,
        sin_diff, cos_diff, tan_diff,
        sin_double, cos_double, tan_double,
        sin_half, cos_half, tan_half,
        plot_unit_circle, visualize_angle_addition,
        explain_angle_addition
    )

Example:
    >>> solve_khan_academy_problem("sin(75)")
    # Shows step-by-step solution using sin(45° + 30°)
"""

from .trig_identities import (
    # Intuition and explanations
    explain_angle_addition,
    explain_unit_circle,
    print_special_angles_table,
    
    # Angle addition identities
    sin_sum,
    cos_sum,
    tan_sum,
    
    # Angle subtraction identities
    sin_diff,
    cos_diff,
    tan_diff,
    
    # Double angle identities
    sin_double,
    cos_double,
    tan_double,
    
    # Half angle identities
    sin_half,
    cos_half,
    tan_half,
    
    # Special angle values
    SPECIAL_ANGLES,
    get_exact_value,
    
    # Problem solving
    solve_khan_academy_problem,
    find_angle_decomposition,
    generate_practice_problems,
    run_practice_session,
)

from .trig_visualizations import (
    plot_unit_circle,
    visualize_angle_addition,
    plot_trig_functions,
    visualize_double_angle,
    visualize_pythagorean_identity,
    visualize_angle_addition_proof,
    create_identity_cheatsheet,
    interactive_angle_explorer,
)

__version__ = "1.0.0"
__author__ = "Machine Learning and AI Repository"

__all__ = [
    # Explanations
    "explain_angle_addition",
    "explain_unit_circle", 
    "print_special_angles_table",
    
    # Addition identities
    "sin_sum",
    "cos_sum",
    "tan_sum",
    
    # Subtraction identities
    "sin_diff",
    "cos_diff",
    "tan_diff",
    
    # Double angle
    "sin_double",
    "cos_double",
    "tan_double",
    
    # Half angle
    "sin_half",
    "cos_half",
    "tan_half",
    
    # Special angles
    "SPECIAL_ANGLES",
    "get_exact_value",
    
    # Problem solving
    "solve_khan_academy_problem",
    "find_angle_decomposition",
    "generate_practice_problems",
    "run_practice_session",
    
    # Visualizations
    "plot_unit_circle",
    "visualize_angle_addition",
    "plot_trig_functions",
    "visualize_double_angle",
    "visualize_pythagorean_identity",
    "visualize_angle_addition_proof",
    "create_identity_cheatsheet",
    "interactive_angle_explorer",
]
