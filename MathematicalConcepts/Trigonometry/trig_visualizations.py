"""
Trigonometry Visualization Module

This module provides visualizations to build intuition for trigonometric concepts:
- Unit circle with angle visualization
- Angle addition formula visual proof
- Comparison of trig functions
- Interactive demonstrations

All plots are designed to help you understand WHY the formulas work,
not just memorize them.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Wedge
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def plot_unit_circle(angles: list = None, show_coordinates: bool = True,
                     figsize: tuple = (10, 10), save_path: str = None):
    """
    Plot the unit circle with optional angle markers.
    
    This visualization helps understand:
    - cos(θ) is the x-coordinate on the unit circle
    - sin(θ) is the y-coordinate on the unit circle
    - The relationship between angles and coordinates
    
    Parameters:
    -----------
    angles : list
        List of angles in degrees to mark on the circle
    show_coordinates : bool
        Whether to show (cos θ, sin θ) labels
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure (optional)
    """
    if angles is None:
        angles = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, label='Unit Circle')
    
    # Draw axes
    ax.axhline(y=0, color='gray', linewidth=1)
    ax.axvline(x=0, color='gray', linewidth=1)
    
    # Mark angles
    colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
    
    for angle, color in zip(angles, colors):
        rad = np.radians(angle)
        x, y = np.cos(rad), np.sin(rad)
        
        # Point on circle
        ax.plot(x, y, 'o', color=color, markersize=10)
        
        # Line from origin to point
        ax.plot([0, x], [0, y], '-', color=color, linewidth=1.5, alpha=0.7)
        
        # Projection lines (to show cos and sin)
        ax.plot([x, x], [0, y], '--', color=color, alpha=0.5, linewidth=1)
        ax.plot([0, x], [y, y], '--', color=color, alpha=0.5, linewidth=1)
        
        # Labels
        label_x = 1.15 * x if abs(x) > 0.1 else (0.15 if x >= 0 else -0.15)
        label_y = 1.15 * y if abs(y) > 0.1 else (0.15 if y >= 0 else -0.15)
        ax.annotate(f'{angle}°', (label_x, label_y), fontsize=10, ha='center', va='center')
        
        if show_coordinates and angle in [0, 30, 45, 60, 90]:
            coord_label = f'({x:.3f}, {y:.3f})'
            ax.annotate(coord_label, (x + 0.1, y - 0.15), fontsize=8, color='gray')
    
    # Add explanation text
    ax.text(0.5, -1.4, 'x = cos(θ)\ny = sin(θ)', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax.transData, ha='center')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('The Unit Circle: Foundation of Trigonometry', fontsize=14, fontweight='bold')
    ax.set_xlabel('cos(θ) →', fontsize=12)
    ax.set_ylabel('sin(θ) →', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def visualize_angle_addition(alpha: float, beta: float, figsize: tuple = (14, 6),
                            save_path: str = None):
    """
    Visual proof of the angle addition formula.
    
    This shows HOW sin(α + β) = sin(α)cos(β) + cos(α)sin(β) works geometrically.
    
    Parameters:
    -----------
    alpha : float
        First angle in degrees
    beta : float
        Second angle in degrees
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    combined_rad = alpha_rad + beta_rad
    
    # ========== LEFT PLOT: Show the two angles ==========
    ax1 = axes[0]
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=1, alpha=0.5)
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)
    
    # Angle α (from x-axis)
    x_alpha = np.cos(alpha_rad)
    y_alpha = np.sin(alpha_rad)
    ax1.plot([0, x_alpha], [0, y_alpha], 'r-', linewidth=2, label=f'α = {alpha}°')
    ax1.plot(x_alpha, y_alpha, 'ro', markersize=10)
    
    # Angle β (from α)
    x_beta = np.cos(combined_rad)
    y_beta = np.sin(combined_rad)
    ax1.plot([0, x_beta], [0, y_beta], 'g-', linewidth=2, label=f'α + β = {alpha + beta}°')
    ax1.plot(x_beta, y_beta, 'go', markersize=10)
    
    # Draw arcs for angles
    arc_alpha = Arc((0, 0), 0.4, 0.4, angle=0, theta1=0, theta2=alpha, color='red', linewidth=2)
    ax1.add_patch(arc_alpha)
    
    arc_beta = Arc((0, 0), 0.6, 0.6, angle=0, theta1=alpha, theta2=alpha + beta, color='green', linewidth=2)
    ax1.add_patch(arc_beta)
    
    # Labels
    ax1.annotate('α', (0.25*np.cos(alpha_rad/2), 0.25*np.sin(alpha_rad/2)), 
                fontsize=12, color='red', fontweight='bold')
    ax1.annotate('β', (0.4*np.cos(alpha_rad + beta_rad/2), 0.4*np.sin(alpha_rad + beta_rad/2)), 
                fontsize=12, color='green', fontweight='bold')
    
    # Show coordinates
    ax1.plot([x_beta, x_beta], [0, y_beta], 'g--', alpha=0.5)
    ax1.plot([0, x_beta], [y_beta, y_beta], 'g--', alpha=0.5)
    
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    ax1.set_title('Angle Addition on the Unit Circle', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ========== RIGHT PLOT: Show the formula breakdown ==========
    ax2 = axes[1]
    
    # Compute components
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    sin_b = np.sin(beta_rad)
    cos_b = np.cos(beta_rad)
    
    sin_sum = sin_a * cos_b + cos_a * sin_b
    
    # Create a bar chart showing the components
    components = ['sin(α)cos(β)', 'cos(α)sin(β)', 'sin(α+β)']
    values = [sin_a * cos_b, cos_a * sin_b, sin_sum]
    colors_bar = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    bars = ax2.bar(components, values, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    # Draw a line showing the sum
    ax2.axhline(y=sin_sum, color='#45b7d1', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f'sin({alpha}° + {beta}°) = sin({alpha}°)cos({beta}°) + cos({alpha}°)sin({beta}°)', 
                  fontsize=11, fontweight='bold')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add the formula as text
    formula_text = f"""
    sin({alpha}°) = {sin_a:.4f}
    cos({alpha}°) = {cos_a:.4f}
    sin({beta}°) = {sin_b:.4f}
    cos({beta}°) = {cos_b:.4f}
    
    Result: {sin_sum:.4f}
    """
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_trig_functions(x_range: tuple = (-2*np.pi, 2*np.pi), 
                       functions: list = None,
                       figsize: tuple = (12, 6),
                       show_special_points: bool = True,
                       save_path: str = None):
    """
    Plot trigonometric functions with key points marked.
    
    Parameters:
    -----------
    x_range : tuple
        Range of x values in radians
    functions : list
        List of functions to plot: 'sin', 'cos', 'tan'
    figsize : tuple
        Figure size
    show_special_points : bool
        Whether to mark special angle points
    save_path : str
        Path to save the figure (optional)
    """
    if functions is None:
        functions = ['sin', 'cos', 'tan']
    
    fig, axes = plt.subplots(len(functions), 1, figsize=figsize, sharex=True)
    if len(functions) == 1:
        axes = [axes]
    
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    colors = {'sin': '#e74c3c', 'cos': '#3498db', 'tan': '#2ecc71'}
    
    for ax, func in zip(axes, functions):
        if func == 'sin':
            y = np.sin(x)
            label = 'sin(x)'
        elif func == 'cos':
            y = np.cos(x)
            label = 'cos(x)'
        elif func == 'tan':
            y = np.tan(x)
            # Handle asymptotes
            y[np.abs(y) > 10] = np.nan
            label = 'tan(x)'
        
        ax.plot(x, y, color=colors[func], linewidth=2, label=label)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        
        # Mark special points
        if show_special_points and func != 'tan':
            special_x = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
            special_labels = ['-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π']
            
            for sx, sl in zip(special_x, special_labels):
                if x_range[0] <= sx <= x_range[1]:
                    if func == 'sin':
                        sy = np.sin(sx)
                    else:
                        sy = np.cos(sx)
                    ax.plot(sx, sy, 'ko', markersize=6)
                    ax.annotate(f'({sl}, {sy:.1f})', (sx, sy), 
                               textcoords="offset points", xytext=(10, 10),
                               fontsize=8)
        
        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2) if func != 'tan' else ax.set_ylim(-5, 5)
        
        # Set x-ticks to show π values
        ax.set_xticks([-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, 
                       np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['-2π', '-3π/2', '-π', '-π/2', '0', 
                            'π/2', 'π', '3π/2', '2π'])
    
    axes[-1].set_xlabel('x (radians)', fontsize=12)
    fig.suptitle('Trigonometric Functions', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def visualize_double_angle(theta: float, figsize: tuple = (14, 5), save_path: str = None):
    """
    Visualize the double angle formulas.
    
    Shows how sin(2θ) and cos(2θ) relate to sin(θ) and cos(θ).
    
    Parameters:
    -----------
    theta : float
        Angle in degrees
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    theta_rad = np.radians(theta)
    
    # Compute values
    sin_t = np.sin(theta_rad)
    cos_t = np.cos(theta_rad)
    sin_2t = np.sin(2 * theta_rad)
    cos_2t = np.cos(2 * theta_rad)
    
    # ========== LEFT: Unit circle with θ and 2θ ==========
    ax1 = axes[0]
    
    # Unit circle
    angles = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(angles), np.sin(angles), 'b-', linewidth=1, alpha=0.5)
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)
    
    # θ
    ax1.plot([0, cos_t], [0, sin_t], 'r-', linewidth=2, label=f'θ = {theta}°')
    ax1.plot(cos_t, sin_t, 'ro', markersize=10)
    
    # 2θ
    ax1.plot([0, np.cos(2*theta_rad)], [0, np.sin(2*theta_rad)], 'g-', linewidth=2, 
            label=f'2θ = {2*theta}°')
    ax1.plot(np.cos(2*theta_rad), np.sin(2*theta_rad), 'go', markersize=10)
    
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    ax1.set_title('θ and 2θ on Unit Circle', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ========== MIDDLE: sin(2θ) = 2sin(θ)cos(θ) ==========
    ax2 = axes[1]
    
    components = ['sin(θ)', 'cos(θ)', '2·sin(θ)·cos(θ)', 'sin(2θ)']
    values = [sin_t, cos_t, 2*sin_t*cos_t, sin_2t]
    colors_bar = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    bars = ax2.bar(components, values, color=colors_bar, edgecolor='black')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', fontsize=9)
    
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.set_title(f'sin(2·{theta}°) = 2·sin({theta}°)·cos({theta}°)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)
    
    # ========== RIGHT: cos(2θ) = cos²(θ) - sin²(θ) ==========
    ax3 = axes[2]
    
    components = ['cos²(θ)', 'sin²(θ)', 'cos²-sin²', 'cos(2θ)']
    values = [cos_t**2, sin_t**2, cos_t**2 - sin_t**2, cos_2t]
    colors_bar = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax3.bar(components, values, color=colors_bar, edgecolor='black')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', fontsize=9)
    
    ax3.axhline(y=0, color='gray', linewidth=0.5)
    ax3.set_title(f'cos(2·{theta}°) = cos²({theta}°) - sin²({theta}°)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def visualize_pythagorean_identity(figsize: tuple = (10, 8), save_path: str = None):
    """
    Visualize the fundamental identity: sin²(θ) + cos²(θ) = 1
    
    This shows WHY this identity is true - it's simply the Pythagorean theorem
    applied to the unit circle!
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ========== LEFT: Geometric proof ==========
    ax1 = axes[0]
    
    theta = 40  # degrees
    theta_rad = np.radians(theta)
    
    # Unit circle
    angles = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(angles), np.sin(angles), 'b-', linewidth=1, alpha=0.5)
    
    # Point on circle
    x = np.cos(theta_rad)
    y = np.sin(theta_rad)
    
    # Draw the right triangle
    ax1.plot([0, x], [0, y], 'r-', linewidth=3, label='Radius = 1')
    ax1.plot([x, x], [0, y], 'g-', linewidth=3, label=f'sin({theta}°) = {y:.3f}')
    ax1.plot([0, x], [0, 0], 'purple', linewidth=3, label=f'cos({theta}°) = {x:.3f}')
    
    ax1.plot(x, y, 'ko', markersize=10)
    ax1.plot(0, 0, 'ko', markersize=6)
    ax1.plot(x, 0, 'ko', markersize=6)
    
    # Right angle marker
    ax1.plot([x-0.05, x-0.05, x], [0, 0.05, 0.05], 'k-', linewidth=1)
    
    # Labels
    ax1.annotate('1', (x/2 - 0.1, y/2 + 0.1), fontsize=14, fontweight='bold', color='red')
    ax1.annotate('sin(θ)', (x + 0.05, y/2), fontsize=12, color='green')
    ax1.annotate('cos(θ)', (x/2, -0.1), fontsize=12, color='purple')
    
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)
    
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.3, 1.3)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left')
    ax1.set_title('Pythagorean Identity: Visual Proof', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add explanation
    explanation = f"""
    By the Pythagorean Theorem:
    
    (cos θ)² + (sin θ)² = 1²
    
    {x:.3f}² + {y:.3f}² = {x**2:.3f} + {y**2:.3f} = {x**2 + y**2:.3f}
    
    This is always true for ANY angle!
    """
    ax1.text(0.02, -0.25, explanation, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== RIGHT: sin²θ + cos²θ for all angles ==========
    ax2 = axes[1]
    
    thetas = np.linspace(0, 360, 361)
    thetas_rad = np.radians(thetas)
    
    sin_sq = np.sin(thetas_rad)**2
    cos_sq = np.cos(thetas_rad)**2
    
    ax2.fill_between(thetas, 0, cos_sq, alpha=0.5, color='blue', label='cos²(θ)')
    ax2.fill_between(thetas, cos_sq, cos_sq + sin_sq, alpha=0.5, color='red', label='sin²(θ)')
    ax2.axhline(y=1, color='green', linewidth=2, linestyle='--', label='Sum = 1')
    
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel('θ (degrees)', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('sin²(θ) + cos²(θ) = 1 for ALL angles', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def visualize_angle_addition_proof(alpha: float = 45, beta: float = 30, 
                                   figsize: tuple = (12, 10), save_path: str = None):
    """
    Complete geometric proof of the angle addition formula.
    
    This visualization shows the classic geometric proof using perpendicular lines.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    combined = alpha_rad + beta_rad
    
    # Scale factor for visibility
    scale = 3
    
    # Point P on unit circle at angle α+β
    P = (scale * np.cos(combined), scale * np.sin(combined))
    
    # Point Q: projection of P onto line at angle α
    # Q is on the line from O at angle α
    # Distance OQ = cos(β) (scaled)
    OQ_length = scale * np.cos(beta_rad)
    Q = (OQ_length * np.cos(alpha_rad), OQ_length * np.sin(alpha_rad))
    
    # Point R: projection of P onto x-axis
    R = (P[0], 0)
    
    # Point S: projection of Q onto x-axis
    S = (Q[0], 0)
    
    # Draw unit circle (scaled)
    angles = np.linspace(0, 2*np.pi, 100)
    ax.plot(scale * np.cos(angles), scale * np.sin(angles), 'b-', linewidth=1, alpha=0.3)
    
    # Draw main lines
    ax.plot([0, P[0]], [0, P[1]], 'g-', linewidth=2, label=f'OP: radius at α+β = {alpha+beta}°')
    ax.plot([0, scale*np.cos(alpha_rad)], [0, scale*np.sin(alpha_rad)], 'r--', 
            linewidth=1.5, alpha=0.7, label=f'Line at α = {alpha}°')
    
    # Draw perpendiculars
    ax.plot([P[0], Q[0]], [P[1], Q[1]], 'purple', linewidth=2, label='PQ ⟂ OQ')
    ax.plot([P[0], R[0]], [P[1], R[1]], 'orange', linewidth=2, label='PR ⟂ x-axis')
    ax.plot([Q[0], S[0]], [Q[1], S[1]], 'cyan', linewidth=2, label='QS ⟂ x-axis')
    
    # Mark points
    points = {'O': (0, 0), 'P': P, 'Q': Q, 'R': R, 'S': S}
    for name, pos in points.items():
        ax.plot(pos[0], pos[1], 'ko', markersize=8)
        offset = (10, 10) if name not in ['R', 'S'] else (10, -15)
        ax.annotate(name, pos, textcoords="offset points", xytext=offset, fontsize=12, fontweight='bold')
    
    # Draw axes
    ax.axhline(y=0, color='gray', linewidth=1)
    ax.axvline(x=0, color='gray', linewidth=1)
    
    # Draw angle arcs
    arc_alpha = Arc((0, 0), 1, 1, angle=0, theta1=0, theta2=alpha, color='red', linewidth=2)
    ax.add_patch(arc_alpha)
    ax.annotate('α', (0.6, 0.3), fontsize=11, color='red')
    
    arc_beta = Arc((0, 0), 1.5, 1.5, angle=0, theta1=alpha, theta2=alpha + beta, color='green', linewidth=2)
    ax.add_patch(arc_beta)
    ax.annotate('β', (0.5, 0.9), fontsize=11, color='green')
    
    # Explanation text
    explanation = f"""
    GEOMETRIC PROOF OF sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
    
    Given: α = {alpha}°, β = {beta}°
    
    From the construction:
    • PR = sin(α + β) · radius = {np.sin(combined)*scale:.3f}
    • OR = cos(α + β) · radius = {np.cos(combined)*scale:.3f}
    
    We can show:
    • PR = QS + (perpendicular component from PQ)
    • PR = sin(α)·OQ + cos(α)·PQ
    • PR = sin(α)·cos(β) + cos(α)·sin(β)   [when radius = 1]
    
    Therefore: sin(α + β) = sin(α)cos(β) + cos(α)sin(β) ✓
    """
    
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Geometric Proof: Angle Addition Formula', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def create_identity_cheatsheet(figsize: tuple = (14, 10), save_path: str = None):
    """
    Create a visual cheatsheet of all major trig identities.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'TRIGONOMETRY IDENTITIES CHEATSHEET', fontsize=18, 
            fontweight='bold', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
    
    # Fundamental Identities
    y = 8.5
    ax.text(0.5, y, 'FUNDAMENTAL IDENTITIES', fontsize=14, fontweight='bold', color='#2c3e50')
    y -= 0.4
    identities = [
        'sin²(θ) + cos²(θ) = 1',
        'tan(θ) = sin(θ)/cos(θ)',
        '1 + tan²(θ) = sec²(θ)',
        '1 + cot²(θ) = csc²(θ)'
    ]
    for identity in identities:
        ax.text(0.7, y, f'• {identity}', fontsize=11, family='monospace')
        y -= 0.35
    
    # Angle Addition
    y -= 0.3
    ax.text(0.5, y, 'ANGLE ADDITION IDENTITIES', fontsize=14, fontweight='bold', color='#27ae60')
    y -= 0.4
    identities = [
        'sin(α + β) = sin(α)cos(β) + cos(α)sin(β)',
        'sin(α - β) = sin(α)cos(β) - cos(α)sin(β)',
        'cos(α + β) = cos(α)cos(β) - sin(α)sin(β)',
        'cos(α - β) = cos(α)cos(β) + sin(α)sin(β)',
        'tan(α + β) = (tan α + tan β)/(1 - tan α tan β)',
        'tan(α - β) = (tan α - tan β)/(1 + tan α tan β)'
    ]
    for identity in identities:
        ax.text(0.7, y, f'• {identity}', fontsize=10, family='monospace')
        y -= 0.35
    
    # Double Angle (right side)
    y_right = 8.5
    ax.text(5.5, y_right, 'DOUBLE ANGLE IDENTITIES', fontsize=14, fontweight='bold', color='#e74c3c')
    y_right -= 0.4
    identities = [
        'sin(2θ) = 2sin(θ)cos(θ)',
        'cos(2θ) = cos²(θ) - sin²(θ)',
        'cos(2θ) = 2cos²(θ) - 1',
        'cos(2θ) = 1 - 2sin²(θ)',
        'tan(2θ) = 2tan(θ)/(1 - tan²θ)'
    ]
    for identity in identities:
        ax.text(5.7, y_right, f'• {identity}', fontsize=10, family='monospace')
        y_right -= 0.35
    
    # Half Angle
    y_right -= 0.3
    ax.text(5.5, y_right, 'HALF ANGLE IDENTITIES', fontsize=14, fontweight='bold', color='#9b59b6')
    y_right -= 0.4
    identities = [
        'sin(θ/2) = ±√[(1 - cos θ)/2]',
        'cos(θ/2) = ±√[(1 + cos θ)/2]',
        'tan(θ/2) = sin θ/(1 + cos θ)',
        'tan(θ/2) = (1 - cos θ)/sin θ'
    ]
    for identity in identities:
        ax.text(5.7, y_right, f'• {identity}', fontsize=10, family='monospace')
        y_right -= 0.35
    
    # Special angles table
    y_table = 3.5
    ax.text(0.5, y_table, 'SPECIAL ANGLES', fontsize=14, fontweight='bold', color='#f39c12')
    
    # Table header
    headers = ['θ', '0°', '30°', '45°', '60°', '90°']
    x_positions = [0.7, 1.5, 2.5, 3.5, 4.5, 5.5]
    y_table -= 0.4
    for header, x_pos in zip(headers, x_positions):
        ax.text(x_pos, y_table, header, fontsize=11, fontweight='bold', ha='center')
    
    # Table rows
    rows = [
        ['sin', '0', '1/2', '√2/2', '√3/2', '1'],
        ['cos', '1', '√3/2', '√2/2', '1/2', '0'],
        ['tan', '0', '√3/3', '1', '√3', 'undef']
    ]
    
    for row in rows:
        y_table -= 0.35
        for val, x_pos in zip(row, x_positions):
            ax.text(x_pos, y_table, val, fontsize=10, ha='center', family='monospace')
    
    # Memory tip
    y_tip = 1.5
    ax.text(0.5, y_tip, 'MEMORY TIP', fontsize=14, fontweight='bold', color='#1abc9c')
    tip_text = """For sin: √0/2, √1/2, √2/2, √3/2, √4/2 → 0, 1/2, √2/2, √3/2, 1
For cos: Go in REVERSE order!
Remember: "All Students Take Calculus" for signs in quadrants (ASTC)"""
    ax.text(0.7, y_tip - 0.3, tip_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def interactive_angle_explorer(save_path: str = None):
    """
    Create a comprehensive visualization showing multiple angles.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== Unit Circle (large) ==========
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
    ax1.axhline(y=0, color='gray', linewidth=1)
    ax1.axvline(x=0, color='gray', linewidth=1)
    
    # Mark special angles
    special_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]
    colors = plt.cm.hsv(np.linspace(0, 0.9, len(special_angles)))
    
    for angle, color in zip(special_angles, colors):
        rad = np.radians(angle)
        x, y = np.cos(rad), np.sin(rad)
        ax1.plot(x, y, 'o', color=color, markersize=10)
        ax1.plot([0, x], [0, y], '-', color=color, linewidth=1, alpha=0.5)
        
        label_x = 1.2 * x if abs(x) > 0.1 else 0.1 * np.sign(x + 0.01)
        label_y = 1.2 * y if abs(y) > 0.1 else 0.1 * np.sign(y + 0.01)
        ax1.annotate(f'{angle}°', (label_x, label_y), fontsize=9, ha='center')
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Unit Circle with Special Angles', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ========== sin and cos graphs ==========
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.linspace(0, 2*np.pi, 100)
    ax2.plot(x, np.sin(x), 'r-', linewidth=2, label='sin(x)')
    ax2.plot(x, np.cos(x), 'b-', linewidth=2, label='cos(x)')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.set_xlabel('x (radians)')
    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax2.legend()
    ax2.set_title('sin(x) and cos(x)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ========== sin² + cos² = 1 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    angles = np.linspace(0, 360, 361)
    sin_sq = np.sin(np.radians(angles))**2
    cos_sq = np.cos(np.radians(angles))**2
    
    ax3.fill_between(angles, 0, cos_sq, alpha=0.6, color='blue', label='cos²(θ)')
    ax3.fill_between(angles, cos_sq, 1, alpha=0.6, color='red', label='sin²(θ)')
    ax3.axhline(y=1, color='green', linestyle='--', linewidth=2)
    ax3.set_xlim(0, 360)
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel('θ (degrees)')
    ax3.legend()
    ax3.set_title('sin²θ + cos²θ = 1', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ========== Double angle demo ==========
    ax4 = fig.add_subplot(gs[1, 1])
    angles = np.linspace(0, 180, 181)
    sin_2x = np.sin(np.radians(2*angles))
    two_sin_cos = 2 * np.sin(np.radians(angles)) * np.cos(np.radians(angles))
    
    ax4.plot(angles, sin_2x, 'r-', linewidth=2, label='sin(2θ)')
    ax4.plot(angles, two_sin_cos, 'b--', linewidth=2, label='2sin(θ)cos(θ)')
    ax4.set_xlabel('θ (degrees)')
    ax4.legend()
    ax4.set_title('Double Angle: sin(2θ) = 2sin(θ)cos(θ)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ========== Angle addition example ==========
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Show sin(45 + 30) = sin(75)
    alpha, beta = 45, 30
    alpha_rad, beta_rad = np.radians(alpha), np.radians(beta)
    
    sin_a, cos_a = np.sin(alpha_rad), np.cos(alpha_rad)
    sin_b, cos_b = np.sin(beta_rad), np.cos(beta_rad)
    
    components = ['sin(α)cos(β)', 'cos(α)sin(β)', 'sin(α+β)']
    values = [sin_a*cos_b, cos_a*sin_b, sin_a*cos_b + cos_a*sin_b]
    colors_bar = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    bars = ax5.bar(components, values, color=colors_bar, edgecolor='black')
    for bar, val in zip(bars, values):
        ax5.annotate(f'{val:.4f}', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center')
    
    ax5.axhline(y=0, color='gray', linewidth=0.5)
    ax5.set_title(f'sin({alpha}° + {beta}°) = {values[2]:.4f}', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=15)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Trigonometry Visual Explorer', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


if __name__ == "__main__":
    print("="*70)
    print("TRIGONOMETRY VISUALIZATION MODULE DEMO")
    print("="*70)
    
    # Demo each visualization
    print("\n1. Unit Circle Visualization")
    plot_unit_circle(angles=[0, 30, 45, 60, 90])
    
    print("\n2. Angle Addition Visualization")
    visualize_angle_addition(45, 30)
    
    print("\n3. Pythagorean Identity")
    visualize_pythagorean_identity()
    
    print("\n4. Trigonometry Cheatsheet")
    create_identity_cheatsheet()
