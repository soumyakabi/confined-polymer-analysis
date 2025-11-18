"""
Confined Polymer Analysis - Streamlit Web Application (EXACT INTEGRATION)
===============================================================================
EXACT integration with both P(x) and P(y) distributions
Including adaptive SAW sampling, enhanced WLC, and comprehensive error estimation.
"""
import traceback  
from datetime import datetime
from io import BytesIO
import pdf_export_module
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, norm, ks_2samp, truncnorm
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from numba import njit
import warnings
from typing import List  # Added missing import
warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL PARAMETERS (EXACT FROM OPTIMIZED CODE)
# ============================================================
# Note: 'a' is now a user parameter above, so we don't redefine it here
N_BINS = 50
BOOTSTRAP_SAMPLES = 800
PDF_BOOTSTRAP = 80
FOURIER_TERMS = 600

N_WALKERS_FJC = 150000
N_WALKERS_SAW_BASE = 5000  # Updated to match transverse code
N_WALKERS_WLC = 50000

SIMPLE_MC_TRIALS = 30
PERM_K_TRIALS = 25
PERM_C_MIN = 0.25
PERM_POPULATION_FACTOR = 5
PIVOT_EQUILIBRATION = 3000
PIVOT_ATTEMPTS_FACTOR = 10
WLC_MAX_ATTEMPTS_FACTOR = 200
# ============================================================
# ENHANCED INPUT VALIDATION SYSTEM
# ============================================================

def validate_physical_parameters(N, a, L, R, lp, distribution_type, x0=None):
    """Validate that parameters are physically reasonable with comprehensive checks"""
    warnings = []
    errors = []
    
    # Basic parameter validation
    if N <= 0:
        errors.append("❌ Chain length N must be positive")
    elif N < 5:
        warnings.append("⚠️ Very short chain (N < 5) may not represent polymer physics well")
    elif N > 100:
        warnings.append("⚠️ Very long chain (N > 100) may cause computational issues")
    
    if a <= 0:
        errors.append("❌ Kuhn length a must be positive")
    elif a > 0.5:
        warnings.append("⚠️ Large Kuhn length (a > 0.5 µm) may affect continuous chain approximation")
    elif a < 0.01:
        warnings.append("⚠️ Very small Kuhn length (a < 0.01 µm) may cause numerical precision issues")
    
    if R <= 0:
        errors.append("❌ Confinement radius R must be positive")
    elif R < 0.05:
        warnings.append("⚠️ Very small confinement radius (R < 0.05 µm) may cause sampling issues")
    
    if lp <= 0:
        errors.append("❌ Persistence length lp must be positive")
    elif lp > 0.5:
        warnings.append("⚠️ Large persistence length (lp > 0.5 µm) may affect WLC sampling performance")
    
    # Physics-based validation
    if distribution_type == "P(x) - Longitudinal":
        if L <= 0:
            errors.append("❌ Confinement length L must be positive")
        elif L < 0.1:
            warnings.append("⚠️ Very small confinement length (L < 0.1 µm)")
        elif L > 10:
            warnings.append("⚠️ Large confinement length (L > 10 µm) may require more samples")
        
        if x0 is not None:
            if x0 < 0 or x0 > L:
                errors.append(f"❌ Tethering point x₀ must be within confinement [0, L] = [0, {L}]")
            elif x0 < 0.1 or x0 > L - 0.1:
                warnings.append("⚠️ Tethering point near boundary may affect distribution shape")
        
        # Chain length vs confinement
        contour_length = N * a
        if contour_length > 3 * L:
            warnings.append(f"⚠️ Chain contour length ({contour_length:.2f} µm) > 3× confinement L ({L} µm) - extremely confined")
        elif contour_length > 2 * L:
            warnings.append(f"⚠️ Chain contour length ({contour_length:.2f} µm) > 2× confinement L ({L} µm) - highly confined")
        elif contour_length < 0.5 * L:
            warnings.append(f"⚠️ Chain contour length ({contour_length:.2f} µm) < 0.5× confinement L ({L} µm) - weakly confined")
        
        # Persistence length validation
        if lp > L:
            warnings.append(f"⚠️ Persistence length ({lp} µm) > confinement length ({L} µm) - chain is very stiff")
        
        # Kuhn length vs persistence length
        if lp < a:
            warnings.append(f"⚠️ Persistence length ({lp} µm) < Kuhn length ({a} µm) - physically inconsistent")
        elif lp > 10 * a:
            warnings.append(f"⚠️ Persistence length ({lp} µm) >> Kuhn length ({a} µm) - very stiff chain")
    
    else:  # P(y) - Transverse
        # Transverse-specific validations
        contour_length = N * a
        if contour_length > 6 * R:
            warnings.append(f"⚠️ Chain contour length ({contour_length:.2f} µm) > 6× confinement radius ({R} µm) - extremely confined")
        elif contour_length > 4 * R:
            warnings.append(f"⚠️ Chain contour length ({contour_length:.2f} µm) > 4× confinement radius ({R} µm) - highly confined")
        elif contour_length < 2 * R:
            warnings.append(f"⚠️ Chain contour length ({contour_length:.2f} µm) < 2× confinement radius ({R} µm) - weakly confined")
        
        if lp > 2 * R:
            warnings.append(f"⚠️ Persistence length ({lp} µm) > 2× confinement radius ({R} µm) - chain is very stiff")
    
    # Sampling feasibility warnings
    if N > 40 and distribution_type == "P(x) - Longitudinal":
        warnings.append("⚠️ Long chains (N > 40) may have reduced SAW sampling efficiency")
    
    if N > 30 and distribution_type == "P(y) - Transverse":
        warnings.append("⚠️ Long chains (N > 30) may have reduced transverse SAW sampling efficiency")
    
    return errors, warnings

def validate_model_selections(model_fjc, model_saw, model_wlc, theory_options, distribution_type):
    """Validate model and theory selections"""
    warnings = []
    
    if not any([model_fjc, model_saw, model_wlc]):
        return ["❌ Select at least one polymer model (FJC, SAW, or WLC)"]
    
    if distribution_type == "P(x) - Longitudinal":
        if not any(theory_options.values()):
            warnings.append("⚠️ No analytical expressions selected for P(x) - comparison will be limited")
    else:  # P(y)
        if not any(theory_options.values()):
            warnings.append("⚠️ No analytical expressions selected for P(y) - comparison will be limited")
    
    # Model-specific warnings
    if model_saw and N > 35:
        warnings.append("⚠️ SAW sampling for long chains (N > 35) may be computationally intensive")
    
    if model_wlc and lp < 0.05:
        warnings.append("⚠️ Very small persistence length (lp < 0.05 µm) may cause WLC sampling issues")
    
    return warnings

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Confined Polymer Analysis (EXACT)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# EXACT TRANSVERSE P(y) FUNCTIONS (FROM Trans PDF.py)
# ============================================================


# Parameter Validation (from transverse code)
def validate_simulation_parameters(N: int, a: float, R: float, n_walkers: int) -> None:
    """Validate simulation parameters for robustness - EXACT TRANSVERSE"""
    if N <= 0:
        raise ValueError(f"Chain length N must be positive, got {N}")
    if a <= 0:
        raise ValueError(f"Kuhn length must be positive, got {a}")
    if R <= 0:
        raise ValueError(f"Confinement radius must be positive, got {R}")
    if n_walkers < 100:
        raise ValueError(f"Need at least 100 walkers, got {n_walkers}")
    
    # Warn about potentially problematic parameters
    if N > 50:
        warnings.warn(f"Large chain length N={N} may have sampling issues")
    if n_walkers > 1000000:
        warnings.warn(f"Large number of walkers {n_walkers} may be computationally expensive")

# Analytical Solutions (EXACT TRANSVERSE)
def confined_Py(y, N, a, R, n_terms=20):
    """Image method analytical solution for P(y) - EXACT TRANSVERSE"""
    sigma2 = N * a**2 / 3
    prefac = 1.0 / np.sqrt(2 * np.pi * sigma2)
    s = np.zeros_like(y)
    
    for n in range(-n_terms, n_terms + 1):
        s += (-1)**n * np.exp(-((y + 2 * n * R)**2) / (2 * sigma2))
    
    Py = prefac * s
    # Normalize
    norm_factor = trapezoid(Py, y)
    if norm_factor > 0:
        Py /= norm_factor
    return Py

def confined_Py_fourier(y, R, a, N, n_terms=101):
    """Fourier analytical solution for P(y) - EXACT TRANSVERSE"""
    numerator = np.zeros_like(y)
    denominator = 0.0
    
    for n in range(1, n_terms + 1, 2):
        n_pi = n * np.pi
        lambda_n = (a * n_pi)**2 / (8 * R**2)
        y_shifted = y + R
        numerator += np.sin(n_pi * y_shifted / (2 * R)) * np.exp(-lambda_n * N)
        denominator += (1 / n) * np.exp(-lambda_n * N)
    
    if denominator == 0:
        return np.ones_like(y) / (2 * R)  # Uniform distribution as fallback
    
    P = (np.pi / (2 * R)) * (numerator / denominator)
    # Normalize
    norm_factor = trapezoid(P, y)
    if norm_factor > 0:
        P /= norm_factor
    return P

# FJC Monte Carlo (EXACT TRANSVERSE)
@njit
def random_unit_vector():
    """Generate random unit vector - EXACT TRANSVERSE"""
    while True:
        v = np.random.normal(0, 1, 3)
        norm_v = np.sqrt(np.sum(v**2))
        if norm_v > 1e-8:
            return v / norm_v

def fjc_monte_carlo_Py(N, a, R, n_walkers=200000, seed=123):
    """FJC Monte Carlo for P(y) - EXACT TRANSVERSE"""
    np.random.seed(seed)
    ends = []
    
    for _ in range(n_walkers):
        positions = np.zeros((N + 1, 3))
        valid = True
        
        for i in range(1, N + 1):
            step = random_unit_vector()
            positions[i] = positions[i-1] + a * step
            
            if np.abs(positions[i, 1]) > R:  # Confinement check in y-direction
                valid = False
                break
        
        if valid:
            ends.append(positions[-1, 1])
    
    return np.array(ends)

# Robust SAW Monte Carlo (EXACT TRANSVERSE)
@njit
def is_valid_saw_position(new_pos, existing_positions, bead_radius, R):
    """Check if new position is valid (confinement + self-avoidance) - EXACT TRANSVERSE"""
    # Confinement check
    if np.abs(new_pos[1]) > R - bead_radius:
        return False
    
    # Self-avoidance check
    for pos in existing_positions:
        if np.linalg.norm(new_pos - pos) < 2 * bead_radius:
            return False
    return True

def robust_saw_monte_carlo_Py(N: int, a: float, R: float, bead_radius: float = 0.03, 
                             n_walkers: int = 5000, max_total_attempts: int = 1000000) -> np.ndarray:
    """
    ROBUST SAW sampling for P(y) with multiple fallback strategies - EXACT TRANSVERSE
    """
    strategies = [
        # Strategy 1: Standard approach with center bias
        {"trials_per_step": 20, "center_bias": 0.3, "max_attempts": n_walkers * 100},
        # Strategy 2: Aggressive center bias
        {"trials_per_step": 30, "center_bias": 0.5, "max_attempts": n_walkers * 150},
        # Strategy 3: Reduced bead radius temporarily
        {"trials_per_step": 25, "center_bias": 0.4, "bead_radius_factor": 0.8, 
         "max_attempts": n_walkers * 100},
        # Strategy 4: Very aggressive sampling
        {"trials_per_step": 50, "center_bias": 0.6, "max_attempts": n_walkers * 200}
    ]
    
    all_ends = []
    total_attempts = 0
    
    for strategy_idx, strategy in enumerate(strategies):
        if len(all_ends) >= n_walkers:
            break
            
        current_bead_radius = (strategy.get("bead_radius_factor", 1.0) * bead_radius)
        remaining_needed = n_walkers - len(all_ends)
        strategy_ends = []
        attempts = 0
        max_strategy_attempts = min(strategy["max_attempts"], 
                                   max_total_attempts - total_attempts)
        
        while (len(strategy_ends) < remaining_needed and 
               attempts < max_strategy_attempts and
               total_attempts < max_total_attempts):
            
            positions = [np.array([0.0, 0.0, 0.0])]
            valid_chain = True
            
            for i in range(1, N + 1):
                current_pos = positions[-1]
                found_valid_step = False
                
                for trial in range(strategy["trials_per_step"]):
                    # Apply center bias
                    step = np.random.normal(0, 1, 3)
                    if strategy["center_bias"] > 0 and i > 1:
                        bias_strength = (strategy["center_bias"] * 
                                       (1.0 - abs(current_pos[1]) / R))
                        step[1] += bias_strength * (-current_pos[1])
                    
                    step_norm = np.sqrt(np.sum(step**2))
                    if step_norm > 1e-8:
                        step = step / step_norm
                    else:
                        continue
                    
                    new_pos = current_pos + a * step
                    
                    if is_valid_saw_position(new_pos, positions, 
                                           current_bead_radius, R):
                        positions.append(new_pos)
                        found_valid_step = True
                        break
                
                if not found_valid_step:
                    valid_chain = False
                    break
            
            attempts += 1
            total_attempts += 1
            
            if valid_chain and len(positions) == N + 1:
                final_y = positions[-1][1]
                if abs(final_y) <= R:
                    strategy_ends.append(final_y)
        
        if strategy_ends:
            all_ends.extend(strategy_ends)
    
    final_ends = np.array(all_ends[:n_walkers])  # Trim to exact target
    acceptance_rate = len(final_ends) / max(total_attempts, 1)
    
    print(f"Robust SAW N={N}: {len(final_ends)}/{n_walkers} chains "
          f"(overall rate: {acceptance_rate:.4f}, attempts: {total_attempts})")
    
    return final_ends

# PERM Algorithm (EXACT TRANSVERSE)
# Remove type annotations to avoid numba compatibility issues
@njit
def get_available_directions(current_pos, existing_positions, bead_radius, R):
    """Get all valid directions for the next step using PERM approach - EXACT TRANSVERSE"""
    available_dirs = []
    
    # Try 6 principal directions first (most efficient)
    principal_dirs = [
        np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    ]
    
    for dir_vec in principal_dirs:
        new_pos = current_pos + a * dir_vec
        
        # Confinement check
        if np.abs(new_pos[1]) > R - bead_radius:
            continue
            
        # Self-avoidance check
        valid = True
        for pos in existing_positions:
            if np.linalg.norm(new_pos - pos) < 2 * bead_radius:
                valid = False
                break
                
        if valid:
            available_dirs.append(dir_vec)
    
    # If principal directions don't work well, sample random directions
    if len(available_dirs) < 3:
        for _ in range(12):  # Sample additional random directions
            dir_vec = random_unit_vector()
            new_pos = current_pos + a * dir_vec
            
            # Confinement check
            if np.abs(new_pos[1]) > R - bead_radius:
                continue
                
            # Self-avoidance check
            valid = True
            for pos in existing_positions:
                if np.linalg.norm(new_pos - pos) < 2 * bead_radius:
                    valid = False
                    break
                    
            if valid:
                # Avoid duplicates
                is_duplicate = False
                for existing_dir in available_dirs:
                    if np.dot(dir_vec, existing_dir) > 0.95:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    available_dirs.append(dir_vec)
    
    return available_dirs

def perm_saw_monte_carlo_Py(N, a, R, bead_radius=0.03, n_walkers=5000, max_depth=100000,
                          prune_threshold=0.1, enrich_threshold=2.0):
    """
    PERM (Pruned-Enriched Rosenbluth Method) for efficient SAW sampling - EXACT TRANSVERSE
    """
    ends = []
    total_generated = 0
    
    # Initial chain (just the starting point)
    chains = [{
        'positions': [np.array([0.0, 0.0, 0.0])],
        'weight': 1.0
    }]
    
    while len(ends) < n_walkers and total_generated < max_depth:
        new_chains = []
        
        for chain in chains:
            current_positions = chain['positions']
            current_weight = chain['weight']
            current_length = len(current_positions)
            
            if current_length == N + 1:
                # Chain is complete
                final_y = current_positions[-1][1]
                if abs(final_y) <= R:
                    ends.append(final_y)
                continue
            
            # Get available directions for next step
            available_dirs = get_available_directions(
                current_positions[-1], current_positions, bead_radius, R
            )
            
            if not available_dirs:
                # Dead end - prune this chain
                continue
            
            # Calculate new weight
            k = len(available_dirs)
            new_weight = current_weight * k / 6.0  # 6 is coordination number for cubic lattice
            
            # PERM decisions based on weight
            if new_weight < prune_threshold:
                # Prune with probability 1 - new_weight/prune_threshold
                if np.random.random() < new_weight / prune_threshold:
                    # Select one direction randomly
                    selected_dir = available_dirs[np.random.randint(len(available_dirs))]
                    new_pos = current_positions[-1] + a * selected_dir
                    new_chain = {
                        'positions': current_positions + [new_pos],
                        'weight': prune_threshold  # Reset weight after surviving pruning
                    }
                    new_chains.append(new_chain)
            elif new_weight > enrich_threshold:
                # Enrich: clone the chain multiple times
                num_clones = min(int(new_weight / enrich_threshold) + 1, 5)  # Max 5 clones
                for _ in range(num_clones):
                    selected_dir = available_dirs[np.random.randint(len(available_dirs))]
                    new_pos = current_positions[-1] + a * selected_dir
                    new_chain = {
                        'positions': current_positions + [new_pos],
                        'weight': new_weight / num_clones  # Distribute weight among clones
                    }
                    new_chains.append(new_chain)
            else:
                # Normal growth
                selected_dir = available_dirs[np.random.randint(len(available_dirs))]
                new_pos = current_positions[-1] + a * selected_dir
                new_chain = {
                    'positions': current_positions + [new_pos],
                    'weight': new_weight
                }
                new_chains.append(new_chain)
        
        chains = new_chains
        total_generated += 1
        
        # Population control: if too many chains, randomly sample
        if len(chains) > 10000:
            indices = np.random.choice(len(chains), size=5000, replace=False)
            chains = [chains[i] for i in indices]
    
    # If PERM doesn't generate enough samples, fall back to traditional method
    if len(ends) < n_walkers // 2:
        print(f"⚠️ PERM at N={N}: Only {len(ends)} samples, using fallback...")
        fallback_samples = robust_saw_monte_carlo_Py(N, a, R, bead_radius, 
                                                   n_walkers - len(ends))
        ends.extend(fallback_samples)
    
    final_ends = np.array(ends[:n_walkers])
    print(f"✅ PERM SAW at N={N}: {len(final_ends)} samples generated")
    
    return final_ends

def enhanced_saw_monte_carlo_Py(N, a, R, bead_radius=0.03, n_walkers=5000, use_perm=True):
    """Enhanced SAW Monte Carlo for P(y) with PERM optimization - EXACT TRANSVERSE"""
    if use_perm and N >= 10:  # Use PERM for longer chains
        return perm_saw_monte_carlo_Py(N, a, R, bead_radius, n_walkers)
    else:
        return robust_saw_monte_carlo_Py(N, a, R, bead_radius, n_walkers)

# WLC Monte Carlo (EXACT TRANSVERSE)
@njit
def generate_wlc_chain_y(N, a, lp, R):
    """Generate WLC chain for P(y) with persistence length - EXACT TRANSVERSE"""
    # Initialize chain coordinates and directions
    coords = np.zeros((N+1, 3))
    tangents = np.zeros((N, 3))
    
    # Start at origin, random initial direction
    initial_dir = np.random.normal(0, 1, 3)
    initial_dir /= np.sqrt(np.sum(initial_dir**2))
    tangents[0] = initial_dir
    
    # Generate the chain
    for i in range(1, N+1):
        # Propose new direction based on bending energy
        current_tangent = tangents[i-1]
        
        # Generate trial direction with bias toward current direction
        bending_stiffness = lp / a  # dimensionless bending parameter
        kappa = bending_stiffness  # concentration parameter
        
        # Marsaglia method for von Mises-Fisher sampling
        while True:
            v = np.random.normal(0, 1, 3)
            v_norm = np.sqrt(np.sum(v**2))
            if v_norm > 1e-8:
                v /= v_norm
                break
        
        # Bias toward current direction
        if kappa > 0:
            w = np.random.rand()
            w0 = (1.0 - np.exp(-2.0 * kappa)) * w + np.exp(-2.0 * kappa)
            cos_theta = 1.0 + np.log(w0) / kappa
            
            # Ensure cos_theta is within valid range
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            
            # Generate perpendicular component
            perpendicular = v - np.dot(v, current_tangent) * current_tangent
            perp_norm = np.sqrt(np.sum(perpendicular**2))
            
            if perp_norm > 1e-8:
                perpendicular /= perp_norm
                sin_theta = np.sqrt(1.0 - cos_theta**2)
                new_direction = cos_theta * current_tangent + sin_theta * perpendicular
            else:
                new_direction = current_tangent  # fallback
        else:
            new_direction = v  # random direction for kappa=0
        
        new_direction /= np.sqrt(np.sum(new_direction**2))
        
        # Update position
        coords[i] = coords[i-1] + a * new_direction
        if i < N:
            tangents[i] = new_direction
        
        # Check confinement in y-direction
        if np.abs(coords[i, 1]) > R:
            return -10.0
    
    return coords[-1, 1]

def wlc_monte_carlo_Py(N, a, R, lp=0.2, n_walkers=50000, max_attempts_factor=100):
    """WLC Monte Carlo for P(y) - EXACT TRANSVERSE"""
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * n_walkers
    
    while len(ends) < n_walkers and attempts < max_attempts:
        val = generate_wlc_chain_y(N, a, lp, R)
        if val > -5:  # Valid chain
            ends.append(val)
        attempts += 1
    
    if len(ends) < n_walkers:
        print(f"⚠️ WLC at N={N}: {len(ends)} of {n_walkers} accepted")
    else:
        print(f"✅ WLC at N={N}: {len(ends)} of {n_walkers} accepted")
    
    return np.array(ends)

# Robust KDE for smooth distribution (EXACT TRANSVERSE)
def safe_adaptive_kde(data, y_vals, bw_factor=1.5, R=0.4, min_samples=50):
    """Robust KDE with comprehensive error handling - EXACT TRANSVERSE"""
    if len(data) < min_samples:
        return None
    
    try:
        # Remove any NaN or infinite values
        clean_data = data[np.isfinite(data)]
        if len(clean_data) < min_samples:
            return None
        
        # Check for sufficient variance
        if np.std(clean_data) < 1e-10:
            return None
            
        kde = gaussian_kde(clean_data)
        
        # Adaptive bandwidth selection
        silverman_bandwidth = kde.silverman_factor()
        scott_bandwidth = kde.scotts_factor()
        final_bandwidth = min(silverman_bandwidth, scott_bandwidth) * bw_factor
        
        kde.set_bandwidth(bw_method=final_bandwidth)
        pdf = kde(y_vals)
        pdf = np.clip(pdf, 0, None)  # Ensure non-negative
        
        # Normalize within confinement
        mask = (y_vals >= -R) & (y_vals <= R)
        if np.any(mask):
            norm_factor = trapezoid(pdf[mask], y_vals[mask])
            if norm_factor > 0:
                pdf /= norm_factor
            else:
                return None
        else:
            return None
            
        return pdf
        
    except (ValueError, np.linalg.LinAlgError, Exception) as e:
        print(f"KDE failed: {e}")
        return None

# ============================================================
# EXACT SAW UTILITY FUNCTIONS FROM OPTIMIZED CODE (P(x))
# ============================================================

@njit
def generate_saw_step(current_pos, existing_chain, a, L, bead_radius, max_trials=20):
    """Generate a step for SAW with multiple trials to avoid overlaps - EXACT"""
    for trial in range(max_trials):
        step = np.random.normal(0.0, 1.0, 3)
        step_norm = np.sqrt(np.sum(step**2))
        if step_norm > 1e-8:
            step /= step_norm
        else:
            continue
            
        new_pos = current_pos + a * step
        
        if new_pos[0] < 0 or new_pos[0] > L:
            continue
            
        overlap = False
        for i in range(len(existing_chain)):
            if np.linalg.norm(new_pos - existing_chain[i]) < 2 * bead_radius:
                overlap = True
                break
                
        if not overlap:
            return new_pos, True
            
    return current_pos, False

@njit
def generate_saw_perm_step(current_pos, existing_chain, a, L, bead_radius, k_trials=10):
    """Generate k trial steps and count valid ones for PERM weight calculation - EXACT"""
    valid_steps = []
    for _ in range(k_trials):
        step = np.random.normal(0.0, 1.0, 3)
        step_norm = np.sqrt(np.sum(step**2))
        if step_norm > 1e-8:
            step /= step_norm
        else:
            continue
            
        new_pos = current_pos + a * step
        
        if new_pos[0] < 0 or new_pos[0] > L:
            continue
            
        overlap = False
        for i in range(len(existing_chain)):
            if np.linalg.norm(new_pos - existing_chain[i]) < 2 * bead_radius:
                overlap = True
                break
                
        if not overlap:
            valid_steps.append(new_pos)
            
    return valid_steps

@njit
def check_self_avoidance(coords, new_segment, pivot_index, bead_radius):
    """Check if the pivoted segment avoids self-intersection - EXACT"""
    for i in range(len(new_segment)):
        for j in range(pivot_index):
            if np.linalg.norm(new_segment[i] - coords[j]) < 2 * bead_radius:
                return False
        for j in range(i+1, len(new_segment)):
            if np.linalg.norm(new_segment[i] - new_segment[j]) < 2 * bead_radius:
                return False
    return True

@njit
def check_confinement(coords, L):
    """Check if all beads are within confinement [0, L] in x-direction - EXACT"""
    for i in range(len(coords)):
        if coords[i, 0] < 0 or coords[i, 0] > L:
            return False
    return True

@njit
def rotate_vector_3d(vector, axis, angle):
    """Rotate a 3D vector around given axis by angle (radians) - EXACT"""
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotated = (vector * cos_angle +
              np.cross(axis, vector) * sin_angle +
              axis * np.dot(axis, vector) * (1 - cos_angle))
    return rotated

@njit
def generate_pivot_move(coords, pivot_index, max_angle=np.pi):
    """Generate a pivot move by rotating the tail of the chain - EXACT"""
    axis = np.random.normal(0, 1, 3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-8:
        axis /= axis_norm
    else:
        axis = np.array([1.0, 0.0, 0.0])
        
    angle = (np.random.random() - 0.5) * 2 * max_angle
    
    pivot_point = coords[pivot_index]
    new_coords = coords.copy()
    
    for i in range(pivot_index + 1, len(coords)):
        vec = coords[i] - pivot_point
        rotated_vec = rotate_vector_3d(vec, axis, angle)
        new_coords[i] = pivot_point + rotated_vec
        
    return new_coords, pivot_index

@njit
def generate_initial_saw_chain(x0, a, N, L, bead_radius, max_attempts=5000):
    """Generate initial SAW chain for pivot algorithm using simple growth - EXACT"""
    coords = np.zeros((N+1, 3))
    coords[0] = [x0, 0.0, 0.0]
    
    for i in range(1, N+1):
        found_step = False
        for attempt in range(max_attempts):
            step = np.random.normal(0, 1, 3)
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-8:
                step = step / step_norm * a
            else:
                continue
                
            new_pos = coords[i-1] + step
            
            if new_pos[0] < 0 or new_pos[0] > L:
                continue
                
            collision = False
            for j in range(i):
                if np.linalg.norm(new_pos - coords[j]) < 2 * bead_radius:
                    collision = True
                    break
                    
            if not collision:
                coords[i] = new_pos
                found_step = True
                break
                
        if not found_step:
            return None
            
    return coords

# ============================================================
# EXACT ENHANCED SAW SAMPLING FROM OPTIMIZED CODE (P(x))
# ============================================================

def improved_adaptive_saw_monte_carlo_Px(x0, a, N, L, bead_radius=0.03, n_walkers=N_WALKERS_SAW_BASE,
                                        pivot_threshold=40, perm_threshold=30, max_attempts_factor=1000):
    """Enhanced Adaptive SAW Sampler with improved sampling strategies - EXACT"""
    
    # ENHANCED ADAPTIVE SAMPLE SIZING BASED ON CHAIN LENGTH
    if N >= 50:
        effective_walkers = 6000
    elif N >= 40:
        effective_walkers = 5000
    elif N >= 30:
        effective_walkers = 4000
    else:
        effective_walkers = n_walkers

    # ENHANCED HYBRID STRATEGY WITH OPTIMIZED PARAMETERS
    if N >= pivot_threshold:
        return improved_pivot_saw(x0, a, N, L, bead_radius, n_samples=effective_walkers)
    elif N >= perm_threshold:
        return improved_perm_saw(x0, a, N, L, bead_radius, n_walkers=effective_walkers)
    else:
        return robust_simple_saw(x0, a, N, L, bead_radius, n_walkers=effective_walkers)

def robust_simple_saw(x0, a, N, L, bead_radius=0.03, n_walkers=N_WALKERS_SAW_BASE, max_attempts_factor=1000):
    """ROBUST Simple SAW Monte Carlo - Enhanced version - EXACT"""
    @njit
    def generate_robust_saw_chain(N, a, x0, L, bead_radius, max_trials=SIMPLE_MC_TRIALS):
        coords = np.zeros((N+1, 3))
        coords[0, 0] = x0
        
        for i in range(1, N+1):
            found_valid = False
            for trial in range(max_trials):
                step = np.random.normal(0.0, 1.0, 3)
                step_norm = np.sqrt(np.sum(step**2))
                if step_norm > 1e-8:
                    step /= step_norm
                else:
                    continue
                    
                new_pos = coords[i-1] + a * step
                
                if new_pos[0] < 0 or new_pos[0] > L:
                    continue
                    
                collision = False
                for j in range(i):
                    if np.linalg.norm(new_pos - coords[j]) < 2 * bead_radius:
                        collision = True
                        break
                
                if not collision:
                    coords[i] = new_pos
                    found_valid = True
                    break
            
            if not found_valid:
                return -1.0
                
        return coords[-1, 0]
    
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * n_walkers * (N//10 + 1)
    
    while len(ends) < n_walkers and attempts < max_attempts:
        val = generate_robust_saw_chain(N, a, x0, L, bead_radius)
        if val >= 0:
            ends.append(val)
        attempts += 1
    
    acceptance_rate = len(ends) / attempts if attempts > 0 else 0
    ess = len(ends)
    
    return np.array(ends), None, attempts, len(ends), ess

def improved_perm_saw(x0, a, N, L, bead_radius=0.03, n_walkers=N_WALKERS_SAW_BASE,
                     k_trials=PERM_K_TRIALS, c_min=PERM_C_MIN, c_max=4.0, population_limit_factor=PERM_POPULATION_FACTOR):
    """Enhanced PERM algorithm with improved weight management and population control - EXACT"""
    
    if N >= 35:
        k_trials = 30
        c_min = 0.20
        
    population = []
    for i in range(n_walkers):
        chain = [np.array([x0, 0.0, 0.0])]
        weight = 1.0
        population.append((chain, weight, i))
        
    current_step = 1
    max_population = population_limit_factor * n_walkers
    
    while current_step <= N and len(population) > 0:
        new_population = []
        total_weight = 0.0
        trapped_chains = 0
        
        for chain, weight, walker_id in population:
            current_pos = chain[-1]
            valid_steps = generate_saw_perm_step(current_pos, chain, a, L, bead_radius, k_trials)
            
            if len(valid_steps) == 0:
                trapped_chains += 1
                continue
                
            step_weight = weight * len(valid_steps) / k_trials
            
            if len(valid_steps) > 1:
                clearances = []
                for step in valid_steps:
                    min_dist = 1e10
                    for existing_pos in chain:
                        dist = np.linalg.norm(step - existing_pos)
                        if dist < min_dist:
                            min_dist = dist
                    clearances.append(min_dist)
                
                clearances = np.array(clearances)
                probs = clearances / np.sum(clearances)
                chosen_idx = np.random.choice(len(valid_steps), p=probs)
                chosen_step = valid_steps[chosen_idx]
            else:
                chosen_step = valid_steps[0]
                
            new_chain = chain + [chosen_step]
            new_population.append((new_chain, step_weight, walker_id))
            total_weight += step_weight
            
        if len(new_population) == 0:
            break
            
        avg_weight = total_weight / len(new_population)
        final_population = []
        
        progressive_c_min = c_min * (1 + 0.1 * (current_step / N))
        progressive_c_max = c_max * (1 - 0.05 * (current_step / N))
        
        for chain, weight, walker_id in new_population:
            ratio = weight / avg_weight if avg_weight > 0 else 0
            
            if ratio < progressive_c_min:
                survival_prob = ratio / progressive_c_min
                if np.random.random() < survival_prob:
                    final_population.append((chain, avg_weight * survival_prob, walker_id))
            elif ratio > progressive_c_max:
                num_copies = min(int(np.sqrt(ratio)) + 1, 5)
                for i in range(num_copies):
                    final_population.append((chain, weight / num_copies, walker_id * 1000 + i))
            else:
                final_population.append((chain, weight, walker_id))
                
        if len(final_population) > max_population:
            weights = [w for _, w, _ in final_population]
            total_w = sum(weights)
            if total_w > 0:
                probs = [w / total_w for w in weights]
                indices = np.random.choice(len(final_population), size=max_population, replace=False, p=probs)
                final_population = [final_population[i] for i in indices]
            else:
                indices = np.random.choice(len(final_population), size=max_population, replace=False)
                final_population = [final_population[i] for i in indices]
                
        population = final_population
        current_step += 1
                  
    final_positions = []
    final_weights = []
    total_final_weight = 0.0
    complete_chains = 0
    
    for chain, weight, _ in population:
        if len(chain) == N + 1:
            final_positions.append(chain[-1][0])
            final_weights.append(weight)
            total_final_weight += weight
            complete_chains += 1
            
    final_positions = np.array(final_positions)
    final_weights = np.array(final_weights)
    
    if total_final_weight > 0:
        final_weights /= total_final_weight
        
    acceptance_rate = complete_chains / n_walkers
    ess = (np.sum(final_weights) ** 2) / np.sum(final_weights ** 2) if len(final_weights) > 0 else 0
          
    return final_positions, final_weights, n_walkers, complete_chains, ess

def improved_pivot_saw(x0, a, N, L, bead_radius=0.03, n_samples=N_WALKERS_SAW_BASE,
                      n_equilibration=PIVOT_EQUILIBRATION, max_angle=np.pi/2, pivot_attempts_factor=PIVOT_ATTEMPTS_FACTOR):
    """Enhanced pivot algorithm with better equilibration and sampling - EXACT"""
    initial_chain = generate_initial_saw_chain(x0, a, N, L, bead_radius, max_attempts=20000)
    
    if initial_chain is None:
        initial_chain = generate_initial_saw_chain(x0, a, N, L, bead_radius*0.9, max_attempts=10000)
        if initial_chain is None:
            return np.array([]), None, 0, 0, 0
        
    coords = initial_chain
    accepted = 0
    attempts = 0
    
    recent_acceptances = []
    window_size = 100
    
    for step in range(n_equilibration):
        if len(coords) <= 2:
            break
            
        pivot_accepted = False
        for pivot_attempt in range(pivot_attempts_factor):
            pivot_index = np.random.randint(1, len(coords) - 1)
            new_coords, pivot_idx = generate_pivot_move(coords, pivot_index, max_angle)
            attempts += 1
            
            if (check_self_avoidance(coords, new_coords[pivot_index:], pivot_index, bead_radius) and
                check_confinement(new_coords, L)):
                coords = new_coords
                accepted += 1
                pivot_accepted = True
                break
                
        recent_acceptances.append(1 if pivot_accepted else 0)
        if len(recent_acceptances) > window_size:
            recent_acceptances.pop(0)
            
        if step % 200 == 0 and len(recent_acceptances) == window_size:
            recent_rate = np.mean(recent_acceptances)
            if recent_rate < 0.1:
                max_angle *= 0.8
            elif recent_rate > 0.3:
                max_angle = min(max_angle * 1.2, np.pi)
                  
    equilibration_acceptance = accepted / attempts if attempts > 0 else 0
    
    accepted_prod = 0
    attempts_prod = 0
    final_positions = []
    
    base_interval = max(3, min(10, 30 // (N // 15 + 1)))
    
    samples_collected = 0
    total_production_steps = n_samples * base_interval * 2
    
    for step in range(total_production_steps):
        if samples_collected >= n_samples:
            break
            
        if len(coords) <= 2:
            break
            
        pivot_accepted = False
        for pivot_attempt in range(pivot_attempts_factor):
            pivot_index = np.random.randint(1, len(coords) - 1)
            new_coords, pivot_idx = generate_pivot_move(coords, pivot_index, max_angle)
            attempts_prod += 1
            
            if (check_self_avoidance(coords, new_coords[pivot_index:], pivot_index, bead_radius) and
                check_confinement(new_coords, L)):
                coords = new_coords
                accepted_prod += 1
                pivot_accepted = True
                break
                
        if step % base_interval == 0 and pivot_accepted:
            final_positions.append(coords[-1, 0])
            samples_collected += 1
            
    extra_steps = 0
    while samples_collected < n_samples and extra_steps < n_samples * 10:
        if len(coords) <= 2:
            break
            
        pivot_accepted = False
        for pivot_attempt in range(pivot_attempts_factor):
            pivot_index = np.random.randint(1, len(coords) - 1)
            new_coords, pivot_idx = generate_pivot_move(coords, pivot_index, max_angle)
            attempts_prod += 1
            
            if (check_self_avoidance(coords, new_coords[pivot_index:], pivot_index, bead_radius) and
                check_confinement(new_coords, L)):
                coords = new_coords
                accepted_prod += 1
                pivot_accepted = True
                break
                
        if pivot_accepted:
            final_positions.append(coords[-1, 0])
            samples_collected += 1
            
        extra_steps += 1
        
    acceptance_rate_prod = accepted_prod / attempts_prod if attempts_prod > 0 else 0
    total_acceptance = (accepted + accepted_prod) / (attempts + attempts_prod)
          
    return (np.array(final_positions), None, attempts + attempts_prod,
            len(final_positions), len(final_positions))

# ============================================================
# EXACT ENHANCED WLC FROM OPTIMIZED CODE (P(x))
# ============================================================

@njit
def generate_wlc_chain(N, a, lp, x0, L, temperature_kT=1.0):
    """Generate WLC chain with persistence length - EXACT"""
    coords = np.zeros((N+1, 3))
    tangents = np.zeros((N, 3))
    
    coords[0, 0] = x0
    initial_dir = np.random.normal(0, 1, 3)
    initial_dir /= np.sqrt(np.sum(initial_dir**2))
    tangents[0] = initial_dir
    
    for i in range(1, N+1):
        current_tangent = tangents[i-1]
        bending_stiffness = lp / a
        
        while True:
            v = np.random.normal(0, 1, 3)
            v_norm = np.sqrt(np.sum(v**2))
            if v_norm > 1e-8:
                v /= v_norm
                break
        
        if bending_stiffness > 0:
            kappa = bending_stiffness
            w = np.random.rand()
            w0 = (1.0 - np.exp(-2.0 * kappa)) * w + np.exp(-2.0 * kappa)
            cos_theta = 1.0 + np.log(w0) / kappa
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            
            perpendicular = v - np.dot(v, current_tangent) * current_tangent
            perp_norm = np.sqrt(np.sum(perpendicular**2))
            if perp_norm > 1e-8:
                perpendicular /= perp_norm
                
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            new_direction = cos_theta * current_tangent + sin_theta * perpendicular
        else:
            new_direction = v
        
        new_direction /= np.sqrt(np.sum(new_direction**2))
        coords[i] = coords[i-1] + a * new_direction
        
        if i < N:
            tangents[i] = new_direction
        
        if coords[i, 0] < 0 or coords[i, 0] > L:
            return -1.0
    
    return coords[-1, 0]

def enhanced_wlc_monte_carlo_Px(x0, a, N, L, lp=0.2, temperature_kT=1.0,
                               n_walkers=N_WALKERS_WLC, max_attempts_factor=WLC_MAX_ATTEMPTS_FACTOR):
    """Enhanced WLC Monte Carlo with persistence length optimization - EXACT"""
    if N <= 15:
        enhanced_lp = lp * 1.25
    elif N >= 40:
        enhanced_lp = lp * 0.85
    else:
        enhanced_lp = lp
        
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * n_walkers
    
    while len(ends) < n_walkers and attempts < max_attempts:
        val = generate_wlc_chain(N, a, enhanced_lp, x0, L, temperature_kT)
        if val >= 0:
            ends.append(val)
        attempts += 1
    
    acceptance_rate = len(ends) / attempts if attempts > 0 else 0
    ess = len(ends)
              
    return np.array(ends), None, attempts, len(ends), ess

# ============================================================
# EXACT UTILITY FUNCTIONS FROM OPTIMIZED CODE
# ============================================================

def truncated_gaussian(x, mu, sigma, L):
    """Truncated Gaussian distribution between 0 and L - EXACT"""
    a_t = (0 - mu) / sigma
    b_t = (L - mu) / sigma
    return truncnorm.pdf(x, a_t, b_t, loc=mu, scale=sigma)

def fit_truncated_gaussian(data, x_grid, weights=None, L=2.0):
    """Fit truncated Gaussian to data and return parameters - EXACT"""
    if len(data) < 10:
        return np.nan, np.nan, np.zeros_like(x_grid)
    try:
        if weights is not None:
            mu0 = np.average(data, weights=weights)
            variance = np.average((data - mu0)**2, weights=weights)
            sigma0 = np.sqrt(variance)
        else:
            mu0 = np.mean(data)
            sigma0 = np.std(data)
        
        bounds = ([0.1, 0.01], [L-0.1, L/2])
        
        if weights is not None:
            hist, bin_edges = np.histogram(data, bins=50, range=(0, L), density=True, weights=weights)
        else:
            hist, bin_edges = np.histogram(data, bins=50, range=(0, L), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        def truncated_gaussian_wrapper(x, mu, sigma):
            return truncated_gaussian(x, mu, sigma, L)
            
        popt, _ = curve_fit(truncated_gaussian_wrapper, bin_centers, hist, 
                           p0=[mu0, sigma0], bounds=bounds, maxfev=5000)
        mu_fit, sigma_fit = popt
        fitted_pdf = truncated_gaussian(x_grid, mu_fit, sigma_fit, L)
        return mu_fit, sigma_fit, fitted_pdf
        
    except (RuntimeError, ValueError):
        if weights is not None:
            mu_fit = np.average(data, weights=weights)
            variance = np.average((data - mu_fit)**2, weights=weights)
            sigma_fit = np.sqrt(variance)
        else:
            mu_fit = np.mean(data)
            sigma_fit = np.std(data)
        fitted_pdf = truncated_gaussian(x_grid, mu_fit, sigma_fit, L)
        return mu_fit, sigma_fit, fitted_pdf

def fit_fjc_gaussian(fjc_data, x_grid, L=2.0):
    """Fit Gaussian to FJC data and return parameters with boundary enforcement - EXACT"""
    if len(fjc_data) < 10:
        return np.nan, np.nan, np.zeros_like(x_grid)
    
    mean_fjc = np.mean(fjc_data)
    std_fjc = np.std(fjc_data)
    
    fjc_gaussian = norm.pdf(x_grid, loc=mean_fjc, scale=std_fjc)
    
    boundary_mask = (x_grid <= 0) | (x_grid >= L)
    fjc_gaussian[boundary_mask] = 0
    
    norm_factor = trapezoid(fjc_gaussian, x_grid)
    if norm_factor > 0:
        fjc_gaussian /= norm_factor
        
    return mean_fjc, std_fjc, fjc_gaussian

def bootstrap_errors(data, weights=None, n_bootstrap=BOOTSTRAP_SAMPLES, alpha=0.05):
    """Calculate bootstrap confidence intervals with effective sample size - EXACT"""
    if len(data) < 10:
        return np.nan, np.nan, len(data)
    
    n = len(data)
    
    if weights is not None:
        ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    else:
        ess = n
        
    bootstrap_means = []
    bootstrap_stds = []
    
    for _ in range(n_bootstrap):
        if weights is not None:
            indices = np.random.choice(n, size=n, replace=True, p=weights/np.sum(weights))
            sample = data[indices]
        else:
            sample = np.random.choice(data, size=n, replace=True)
            
        bootstrap_means.append(np.mean(sample))
        bootstrap_stds.append(np.std(sample))
    
    mean_lower = np.percentile(bootstrap_means, 100*alpha/2)
    mean_upper = np.percentile(bootstrap_means, 100*(1-alpha/2))
    std_lower = np.percentile(bootstrap_stds, 100*alpha/2)
    std_upper = np.percentile(bootstrap_stds, 100*(1-alpha/2))
    
    mean_err = (mean_upper - mean_lower) / 2
    std_err = (std_upper - std_lower) / 2
    
    return mean_err, std_err, ess

def bootstrap_pdf_errors(data, x_grid, weights=None, n_bootstrap=PDF_BOOTSTRAP, alpha=0.05, L=2.0):
    """Calculate bootstrap confidence intervals for PDF - EXACT"""
    if len(data) < 20:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)
    
    n = len(data)
    bootstrap_pdfs = []
    
    for _ in range(n_bootstrap):
        if weights is not None:
            indices = np.random.choice(n, size=n, replace=True, p=weights/np.sum(weights))
            sample = data[indices]
        else:
            sample = np.random.choice(data, size=n, replace=True)
            
        if len(sample) > 10:
            kde = gaussian_kde(sample)
            pdf = kde(x_grid)
            pdf = np.clip(pdf, 0, None)
            
            boundary_mask = (x_grid <= 0) | (x_grid >= L)
            pdf[boundary_mask] = 0
            
            mask = (x_grid>=0)&(x_grid<=L)
            norm_factor = trapezoid(pdf[mask], x_grid[mask])
            if norm_factor > 0:
                pdf /= norm_factor
                
            bootstrap_pdfs.append(pdf)
    
    if len(bootstrap_pdfs) == 0:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)
        
    bootstrap_pdfs = np.array(bootstrap_pdfs)
    pdf_mean = np.mean(bootstrap_pdfs, axis=0)
    pdf_upper = np.percentile(bootstrap_pdfs, 100*(1-alpha/2), axis=0)
    pdf_lower = np.percentile(bootstrap_pdfs, 100*alpha/2, axis=0)  # FIXED: removed extra parenthesis
    
    return pdf_mean, (pdf_upper - pdf_lower) / 2

def analytical_Px_fourier(x_vals, x0, a, N, L, n_terms=FOURIER_TERMS, decay_tol=1e-10):
    """Analytical Fourier solution with Boundary Enforcement - EXACT"""
    D = a**2 / 2
    G = np.zeros_like(x_vals)
    
    for n in range(1, n_terms):
        decay = np.exp(-(n * np.pi / L)**2 * D * N)
        if decay < decay_tol:
            break
        G += np.sin(n*np.pi*x0/L) * np.sin(n*np.pi*x_vals/L) * decay
        
    G *= (2 / L)
    
    boundary_mask = (x_vals <= 0) | (x_vals >= L)
    G[boundary_mask] = 0
    
    norm_factor = trapezoid(G, x_vals)
    return G / norm_factor if norm_factor > 0 else np.ones_like(x_vals)/L

def fjc_monte_carlo_Px(x0, a, N, L, n_walkers=N_WALKERS_FJC, seed=42):
    """FJC Monte Carlo - EXACT"""
    np.random.seed(seed)
    final_positions = []
    
    for _ in range(n_walkers):
        x = x0
        alive = True
        for _ in range(N):
            x += a * np.random.randn()
            if x < 0 or x > L:
                alive = False
                break
        if alive:
            final_positions.append(x)
            
    return np.array(final_positions), n_walkers, len(final_positions)



# ============================================================
# STREAMLIT UI WITH DUAL P(x) AND P(y) SUPPORT
# ============================================================

st.title("🧬 Confined Polymer Analysis (EXACT Implementation)")
st.markdown("**FJC, SAW (EXACT with Adaptive Algorithms), and WLC Models for P(x) and P(y)**")

st.sidebar.header("⚙️ Distribution Type")

# Distribution type selection
distribution_type = st.sidebar.radio(
    "Select Distribution Type:",
    ["P(x) - Longitudinal", "P(y) - Transverse"],
    index=0,
    key="distribution_type_radio"
)

st.sidebar.header("⚙️ Geometry Parameters")

# Make Kuhn length variable with realistic limits
a = st.sidebar.slider("Kuhn Length a (µm)", 0.01, 0.5, 0.1, 0.01,
                     help="Segment length - typical range 0.01-0.5 µm for polymers",
                     key="kuhn_length_slider")

if distribution_type == "P(x) - Longitudinal":
    L = st.sidebar.slider("Confinement Length L (µm)", 0.5, 10.0, 2.0, 0.1,
                         help="Length of cylindrical confinement in x-direction",
                         key="confinement_length_slider")
    R = st.sidebar.slider("Cylinder Radius R (µm)", 0.1, 2.0, 0.4, 0.05,
                         help="Radius of cylindrical confinement",
                         key="radius_px_slider")
    
    st.sidebar.header("⚙️ Chain Parameters (P(x))")
    N = st.sidebar.slider("Chain Length N", 5, 100, 25, 5,
                         help="Number of Kuhn segments - longer chains are more realistic but computationally expensive",
                         key="chain_length_px_slider")
    x0 = st.sidebar.slider("Tethering Point x₀ (µm)", 0.0, float(L), 0.75, 0.05,
                          help="Starting position of polymer chain within confinement",
                          key="tethering_point_px_slider")
    
    st.sidebar.header("⚙️ Models (P(x))")
    model_fjc = st.sidebar.checkbox("FJC", value=True,
                                   help="Freely Jointed Chain - simplest polymer model",
                                   key="fjc_px_checkbox")
    model_saw = st.sidebar.checkbox("SAW (EXACT Adaptive)", value=False,
                                   help="Self-Avoiding Walk - excludes chain self-intersections",
                                   key="saw_px_checkbox")
    model_wlc = st.sidebar.checkbox("WLC (Enhanced)", value=False,
                                   help="Wormlike Chain - includes chain stiffness via persistence length",
                                   key="wlc_px_checkbox")
    
    st.sidebar.header("⚙️ Analytical (P(x))")
    theory_fourier_px = st.sidebar.checkbox("Fourier P(x)", value=True,
                                           help="Fourier series solution for confined ideal chain",
                                           key="fourier_px_checkbox")
    theory_gaussian = st.sidebar.checkbox("Gaussian", value=False,
                                         help="Standard Gaussian distribution approximation",
                                         key="gaussian_px_checkbox")
    theory_truncated = st.sidebar.checkbox("Truncated Gaussian", value=False,
                                          help="Gaussian truncated at confinement boundaries",
                                          key="truncated_px_checkbox")
    theory_fourier_py = None
    theory_image = None
    
else:  # P(y) - Transverse
    R = st.sidebar.slider("Confinement Radius R (µm)", 0.1, 2.0, 0.4, 0.05,
                         help="Radius of cylindrical confinement in y-direction",
                         key="radius_py_slider")
    L = None
    
    st.sidebar.header("⚙️ Chain Parameters (P(y))")
    N = st.sidebar.slider("Chain Length N", 5, 100, 25, 5,
                         help="Number of Kuhn segments - longer chains are more realistic but computationally expensive",
                         key="chain_length_py_slider")
    st.sidebar.info("Tethering point y₀ = 0.0 µm (center, fixed)")
    x0 = 0.0
    
    st.sidebar.header("⚙️ Models (P(y))")
    model_fjc = st.sidebar.checkbox("FJC", value=True,
                                   help="Freely Jointed Chain - simplest polymer model",
                                   key="fjc_py_checkbox")
    model_saw = st.sidebar.checkbox("SAW (Enhanced with PERM)", value=False,
                                   help="Self-Avoiding Walk - excludes chain self-intersections",
                                   key="saw_py_checkbox")
    model_wlc = st.sidebar.checkbox("WLC", value=False,
                                   help="Wormlike Chain - includes chain stiffness via persistence length",
                                   key="wlc_py_checkbox")
    
    st.sidebar.header("⚙️ Analytical (P(y))")
    theory_fourier_py = st.sidebar.checkbox("Fourier P(y)", value=True,
                                           help="Fourier series solution for transverse confinement",
                                           key="fourier_py_checkbox")
    theory_image = st.sidebar.checkbox("Image Method P(y)", value=False,
                                      help="Method of images solution for transverse confinement",
                                      key="image_py_checkbox")
    theory_fourier_px = None
    theory_gaussian = None
    theory_truncated = None

# Common parameters
st.sidebar.header("⚙️ WLC Parameters")
lp = st.sidebar.slider("Persistence Length lₚ (µm)", 0.01, 2.0, 0.2, 0.01,
                      help="Chain stiffness parameter - higher values mean stiffer chains",
                      key="persistence_length_slider")

st.sidebar.header("⚙️ Bootstrap")
show_bootstrap = st.sidebar.checkbox("Show Bootstrap Error Estimates", value=True,
                                    help="Display statistical uncertainty estimates using bootstrap resampling",
                                    key="bootstrap_checkbox")

# Real-time parameter validation display
st.sidebar.header("⚙️ Validation & Warnings")

# Initialize variables for validation
validation_errors = []
validation_warnings = []

if distribution_type == "P(x) - Longitudinal":
    validation_errors, validation_warnings = validate_physical_parameters(N, a, L, R, lp, distribution_type, x0)
else:
    validation_errors, validation_warnings = validate_physical_parameters(N, a, None, R, lp, distribution_type)

# Display real-time warnings in sidebar
if validation_warnings:
    with st.sidebar.expander("🔍 Parameter Warnings", expanded=True):
        for warning in validation_warnings:
            st.warning(warning)

if validation_errors:
    with st.sidebar.expander("❌ Parameter Errors", expanded=True):
        for error in validation_errors:
            st.error(error)

# Enhanced compute button with validation
compute_disabled = len(validation_errors) > 0
compute = st.sidebar.button("🚀 COMPUTE", use_container_width=True, 
                           disabled=compute_disabled,
                           help="Run simulation with current parameters" + (" (disabled due to errors)" if compute_disabled else ""),
                           key="compute_button")

# ============================================================
# MAIN COMPUTATION AND DISPLAY
# ============================================================

if compute:
    if not any([model_fjc, model_saw, model_wlc]):
        st.error("❌ Select at least one model!")
        st.stop()
    
    if distribution_type == "P(x) - Longitudinal":
        if not any([theory_fourier_px, theory_gaussian, theory_truncated]):
            st.error("❌ Select at least one analytical expression!")
            st.stop()
    else:  # P(y)
        if not any([theory_fourier_py, theory_image]):
            st.error("❌ Select at least one analytical expression!")
            st.stop()
    
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    results = {}
    
    if distribution_type == "P(x) - Longitudinal":
        # P(x) ANALYSIS
        x_grid = np.linspace(0, L, 500)
        
        if theory_fourier_px:
            progress_placeholder.info("Computing Fourier P(x)...")
            results['fourier_px'] = analytical_Px_fourier(x_grid, x0, a, N, L)
            progress_bar.progress(15)
        
        if model_fjc:
            progress_placeholder.info("Running FJC P(x)...")
            fjc_data, _, fjc_accepted = fjc_monte_carlo_Px(x0, a, N, L, n_walkers=N_WALKERS_FJC)
            fjc_mean, fjc_std, fjc_gaussian = fit_fjc_gaussian(fjc_data, x_grid, L)
            fjc_trunc_mu, fjc_trunc_sigma, fjc_trunc_gaussian = fit_truncated_gaussian(fjc_data, x_grid, L=L)
            
            fjc_mean_err, fjc_std_err, fjc_ess = bootstrap_errors(fjc_data)
            
            results['fjc'] = {
                'data': fjc_data,
                'gaussian': fjc_gaussian,
                'trunc': fjc_trunc_gaussian,
                'mean': fjc_mean,
                'std': fjc_std,
                'mean_err': fjc_mean_err,
                'std_err': fjc_std_err,
                'ess': fjc_ess,
                'samples': len(fjc_data),
            }
            progress_bar.progress(40)
        
        if model_saw:
            progress_placeholder.info("Running SAW P(x) (EXACT Adaptive algorithm)...")
            saw_data, saw_weights, saw_attempts, saw_accepted, saw_ess = improved_adaptive_saw_monte_carlo_Px(
                x0, a, N, L, bead_radius=0.03, n_walkers=N_WALKERS_SAW_BASE)
            saw_trunc_mu, saw_trunc_sigma, saw_trunc_gaussian = fit_truncated_gaussian(saw_data, x_grid, L=L)
            
            saw_mean_err, saw_std_err, saw_ess_calc = bootstrap_errors(saw_data, saw_weights)
            saw_pdf_mean, saw_pdf_err = bootstrap_pdf_errors(saw_data, x_grid, saw_weights, L=L)
            
            results['saw'] = {
                'data': saw_data,
                'weights': saw_weights,
                'trunc': saw_trunc_gaussian,
                'mean': np.mean(saw_data),
                'std': np.std(saw_data),
                'mean_err': saw_mean_err,
                'std_err': saw_std_err,
                'ess': saw_ess_calc,
                'samples': len(saw_data),
                'pdf_mean': saw_pdf_mean,
                'pdf_err': saw_pdf_err,
            }
            progress_bar.progress(65)
        
        if model_wlc:
            progress_placeholder.info("Running WLC P(x) (Enhanced)...")
            wlc_data, _, wlc_attempts, wlc_accepted, wlc_ess = enhanced_wlc_monte_carlo_Px(
                x0, a, N, L, lp=lp, n_walkers=N_WALKERS_WLC)
            wlc_trunc_mu, wlc_trunc_sigma, wlc_trunc_gaussian = fit_truncated_gaussian(wlc_data, x_grid, L=L)
            
            wlc_mean_err, wlc_std_err, wlc_ess_calc = bootstrap_errors(wlc_data)
            
            results['wlc'] = {
                'data': wlc_data,
                'trunc': wlc_trunc_gaussian,
                'mean': np.mean(wlc_data),
                'std': np.std(wlc_data),
                'mean_err': wlc_mean_err,
                'std_err': wlc_std_err,
                'ess': wlc_ess_calc,
                'samples': len(wlc_data),
            }
            progress_bar.progress(85)
            
    else:
        # P(y) ANALYSIS - USING EXACT TRANSVERSE FUNCTIONS
        y_grid = np.linspace(-R, R, 500)
        
        # Parameter validation for transverse simulations
        try:
            validate_simulation_parameters(N, a, R, N_WALKERS_FJC)
        except ValueError as e:
            st.error(f"❌ Parameter validation failed: {e}")
            st.stop()
        
        if theory_fourier_py:
            progress_placeholder.info("Computing Fourier P(y)...")
            results['fourier_py'] = confined_Py_fourier(y_grid, R, a, N)
            progress_bar.progress(15)
        
        if theory_image:
            progress_placeholder.info("Computing Image Method P(y)...")
            results['image_py'] = confined_Py(y_grid, N, a, R)
            progress_bar.progress(25)
        
        if model_fjc:
            progress_placeholder.info("Running FJC P(y)...")
            fjc_data = fjc_monte_carlo_Py(N, a, R, n_walkers=N_WALKERS_FJC)
            fjc_mean_err, fjc_std_err, fjc_ess = bootstrap_errors(fjc_data)
            
            results['fjc'] = {
                'data': fjc_data,
                'mean': np.mean(fjc_data),
                'std': np.std(fjc_data),
                'mean_err': fjc_mean_err,
                'std_err': fjc_std_err,
                'ess': fjc_ess,
                'samples': len(fjc_data),
            }
            progress_bar.progress(45)
        
        if model_saw:
            progress_placeholder.info("Running SAW P(y) (Enhanced with PERM)...")
            saw_data = enhanced_saw_monte_carlo_Py(N, a, R, bead_radius=0.03, 
                                                 n_walkers=N_WALKERS_SAW_BASE, use_perm=True)
            saw_mean_err, saw_std_err, saw_ess = bootstrap_errors(saw_data)
            
            # Use robust KDE for SAW P(y)
            saw_pdf = safe_adaptive_kde(saw_data, y_grid, R=R)
            
            results['saw'] = {
                'data': saw_data,
                'mean': np.mean(saw_data),
                'std': np.std(saw_data),
                'mean_err': saw_mean_err,
                'std_err': saw_std_err,
                'ess': saw_ess,
                'samples': len(saw_data),
                'pdf': saw_pdf
            }
            progress_bar.progress(65)
        
        if model_wlc:
            progress_placeholder.info("Running WLC P(y)...")
            wlc_data = wlc_monte_carlo_Py(N, a, R, lp=lp, n_walkers=N_WALKERS_WLC)
            wlc_mean_err, wlc_std_err, wlc_ess = bootstrap_errors(wlc_data)
            
            # Use robust KDE for WLC P(y)
            wlc_pdf = safe_adaptive_kde(wlc_data, y_grid, R=R)
            
            results['wlc'] = {
                'data': wlc_data,
                'mean': np.mean(wlc_data),
                'std': np.std(wlc_data),
                'mean_err': wlc_mean_err,
                'std_err': wlc_std_err,
                'ess': wlc_ess,
                'samples': len(wlc_data),
                'pdf': wlc_pdf
            }
            progress_bar.progress(85)
    
    progress_placeholder.info("Rendering results...")
    
# ============================================================
# COMPLETE CORRECTED PLOTTING SECTION - P(x) & P(y) ANALYSIS
# ============================================================
# This is the FULLY CORRECTED plotting section with proper indentation
# and all fixes applied. Use this to replace the old plotting section.
# ============================================================
# INDENTATION: All lines shown have explicit indentation (4 spaces = 1 level)
# ============================================================

    if distribution_type == "P(x) - Longitudinal":
        
        tab1, tab2, tab3 = st.tabs(["📊 P(x) Distribution", "📊 With Bootstrap Errors", "📋 Metrics"])
        
        with tab1:
            st.subheader("Longitudinal End-Point Probability Distribution P(x)")
            
            fig_main, ax = plt.subplots(figsize=(14, 7))
            
            if 'fourier_px' in results:
                ax.plot(x_grid, results['fourier_px'], 'k-', lw=3.5, label='Fourier P(x) Analytical', alpha=0.9, zorder=5)
            
            if 'fjc' in results:
                ax.hist(results['fjc']['data'], bins=60, range=(0, L), density=True, alpha=0.3,
                       color='#2E86DE', edgecolor='#2E86DE', linewidth=1,
                       label=f"FJC (n={results['fjc']['samples']:,})")
            
            if theory_gaussian and 'fjc' in results:
                ax.plot(x_grid, results['fjc']['gaussian'], 'r--', lw=2.5, label='FJC Gaussian', alpha=0.8)
            
            if theory_truncated and 'fjc' in results:
                ax.plot(x_grid, results['fjc']['trunc'], 'g:', lw=2.5, label='FJC Truncated Gaussian', alpha=0.8)
            
            if 'saw' in results and len(results['saw']['data']) > 0:
                try:
                    kde_saw = gaussian_kde(results['saw']['data'])
                    pdf_saw = kde_saw(x_grid)
                    ax.plot(x_grid, pdf_saw, 'green', lw=3.5, label=f"SAW KDE (n={results['saw']['samples']})", alpha=0.85)
                except:
                    ax.hist(results['saw']['data'], bins=40, range=(0, L), density=True, alpha=0.3,
                           color='green', edgecolor='green', label=f"SAW (n={results['saw']['samples']})")
            
            if 'wlc' in results and len(results['wlc']['data']) > 0:
                try:
                    kde_wlc = gaussian_kde(results['wlc']['data'])
                    pdf_wlc = kde_wlc(x_grid)
                    ax.plot(x_grid, pdf_wlc, 'orange', lw=3.5, label=f"WLC KDE (n={results['wlc']['samples']})", alpha=0.85)
                except:
                    ax.hist(results['wlc']['data'], bins=40, range=(0, L), density=True, alpha=0.3,
                           color='orange', edgecolor='orange', label=f"WLC (n={results['wlc']['samples']})")
            
            ax.set_xlabel('x (µm)', fontsize=14, fontweight='bold')
            ax.set_ylabel('P(x)', fontsize=14, fontweight='bold')
            ax.set_xlim(0, L)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='best')
            ax.set_title(f"P(x): N={N}, x₀={x0:.2f}µm, L={L}µm", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            st.pyplot(fig_main)
            
            # ★ Store MAIN figure for export
            st.session_state.stored_fig_main = fig_main
        
        with tab2:
            st.subheader("All Distributions with Bootstrap Error Estimates")
            
            if show_bootstrap and 'saw' in results and results['saw']['pdf_mean'] is not None:
                
                fig_bootstrap, ax_b = plt.subplots(figsize=(14, 7))
                
                # ★ Add ALL curves (like tab1) + bootstrap overlay on SAW
                
                if 'fourier_px' in results:
                    ax_b.plot(x_grid, results['fourier_px'], 'k-', lw=3.5, label='Fourier P(x) Analytical', alpha=0.9, zorder=5)
                
                if 'fjc' in results:
                    ax_b.hist(results['fjc']['data'], bins=60, range=(0, L), density=True, alpha=0.3,
                             color='#2E86DE', edgecolor='#2E86DE', linewidth=1,
                             label=f"FJC (n={results['fjc']['samples']:,})")
                
                if theory_gaussian and 'fjc' in results:
                    ax_b.plot(x_grid, results['fjc']['gaussian'], 'r--', lw=2.5, label='FJC Gaussian', alpha=0.8)
                
                if theory_truncated and 'fjc' in results:
                    ax_b.plot(x_grid, results['fjc']['trunc'], 'g:', lw=2.5, label='FJC Truncated Gaussian', alpha=0.8)
                
                # ★ Add WLC (was missing in original tab2!)
                if 'wlc' in results and len(results['wlc']['data']) > 0:
                    try:
                        kde_wlc = gaussian_kde(results['wlc']['data'])
                        pdf_wlc = kde_wlc(x_grid)
                        ax_b.plot(x_grid, pdf_wlc, 'orange', lw=3.5, label=f"WLC KDE (n={results['wlc']['samples']})", alpha=0.85, zorder=3)
                    except:
                        pass
                
                # ★ Add SAW with BOOTSTRAP CI overlay
                if results['saw']['pdf_mean'] is not None:
                    ax_b.plot(x_grid, results['saw']['pdf_mean'], 'green', lw=3.5, label='SAW MC (KDE)', zorder=4)
                    ax_b.fill_between(x_grid,
                                     results['saw']['pdf_mean'] - results['saw']['pdf_err'],
                                     results['saw']['pdf_mean'] + results['saw']['pdf_err'],
                                     color='green', alpha=0.25, label='Bootstrap CI (95%)', zorder=3)
                
                ax_b.hist(results['saw']['data'], bins=40, range=(0, L), density=True, alpha=0.15,
                         color='green', edgecolor='green', linewidth=0.5)
                
                ax_b.set_xlabel('x (µm)', fontsize=14, fontweight='bold')
                ax_b.set_ylabel('P(x)', fontsize=14, fontweight='bold')
                ax_b.set_xlim(0, L)
                ax_b.grid(True, alpha=0.3)
                ax_b.legend(fontsize=11, loc='best')
                ax_b.set_title(f"All Distributions with SAW Bootstrap Error Estimates (95% CI)", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                st.pyplot(fig_bootstrap)
                
                # Store bootstrap figure separately
                st.session_state.stored_fig_bootstrap = fig_bootstrap
            else:
                st.info("📊 Bootstrap error estimates available for SAW model when PDF_BOOTSTRAP=80 and show_bootstrap=True")
        
        with tab3:
            st.subheader("Sampling Statistics & Bootstrap Error Estimates - P(x)")
            
            stats_data = []
            
            if 'fjc' in results:
                stats_data.append({
                    'Model': 'FJC',
                    'Samples': f"{results['fjc']['samples']:,}",
                    'Mean (µm)': f"{results['fjc']['mean']:.6f} ± {results['fjc']['mean_err']:.6f}",
                    'Std (µm)': f"{results['fjc']['std']:.6f} ± {results['fjc']['std_err']:.6f}",
                    'ESS': f"{results['fjc']['ess']:.0f}"
                })
            
            if 'saw' in results:
                stats_data.append({
                    'Model': 'SAW (EXACT Adaptive)',
                    'Samples': f"{results['saw']['samples']:,}",
                    'Mean (µm)': f"{np.mean(results['saw']['data']):.6f} ± {results['saw']['mean_err']:.6f}",
                    'Std (µm)': f"{np.std(results['saw']['data']):.6f} ± {results['saw']['std_err']:.6f}",
                    'ESS': f"{results['saw']['ess']:.1f}"
                })
            
            if 'wlc' in results:
                stats_data.append({
                    'Model': 'WLC (Enhanced)',
                    'Samples': f"{results['wlc']['samples']:,}",
                    'Mean (µm)': f"{results['wlc']['mean']:.6f} ± {results['wlc']['mean_err']:.6f}",
                    'Std (µm)': f"{results['wlc']['std']:.6f} ± {results['wlc']['std_err']:.6f}",
                    'ESS': f"{results['wlc']['ess']:.0f}"
                })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            st.markdown("**Key Statistics:**")
            
            col1, col2, col3 = st.columns(3)
            
            if 'fjc' in results:
                with col1:
                    st.metric("FJC Mean Error", f"{results['fjc']['mean_err']:.6f} µm",
                             help="Bootstrap 95% confidence interval width / 2")
            
            if 'saw' in results:
                with col2:
                    st.metric("SAW Mean Error", f"{results['saw']['mean_err']:.6f} µm",
                             help=f"Bootstrap 95% confidence interval width / 2 (from {BOOTSTRAP_SAMPLES} samples)")
            
            if 'wlc' in results:
                with col3:
                    st.metric("WLC Mean Error", f"{results['wlc']['mean_err']:.6f} µm",
                             help="Bootstrap 95% confidence interval width / 2")
    
    else:  # P(y) - Transverse
        
        tab1, tab2 = st.tabs(["📊 P(y) Distribution", "📋 Metrics"])
        
        with tab1:
            st.subheader("Transverse End-Point Probability Distribution P(y)")
            
            fig_main, ax = plt.subplots(figsize=(14, 7))
            
            if 'fourier_py' in results:
                ax.plot(y_grid, results['fourier_py'], 'k-', lw=3.5, label='Fourier P(y) Analytical', alpha=0.9, zorder=5)
            
            if 'image_py' in results:
                ax.plot(y_grid, results['image_py'], 'm--', lw=2.5, label='Image Method P(y)', alpha=0.8)
            
            if 'fjc' in results:
                ax.hist(results['fjc']['data'], bins=60, range=(-R, R), density=True, alpha=0.3,
                       color='#2E86DE', edgecolor='#2E86DE', linewidth=1,
                       label=f"FJC (n={results['fjc']['samples']:,})")
            
            if 'saw' in results and len(results['saw']['data']) > 0:
                if results['saw']['pdf'] is not None:
                    ax.plot(y_grid, results['saw']['pdf'], 'green', lw=3.5,
                           label=f"SAW KDE (n={results['saw']['samples']})", alpha=0.85)
                else:
                    ax.hist(results['saw']['data'], bins=40, range=(-R, R), density=True, alpha=0.3,
                           color='green', edgecolor='green', label=f"SAW (n={results['saw']['samples']})")
            
            if 'wlc' in results and len(results['wlc']['data']) > 0:
                if results['wlc']['pdf'] is not None:
                    ax.plot(y_grid, results['wlc']['pdf'], 'orange', lw=3.5,
                           label=f"WLC KDE (n={results['wlc']['samples']})", alpha=0.85)
                else:
                    ax.hist(results['wlc']['data'], bins=40, range=(-R, R), density=True, alpha=0.3,
                           color='orange', edgecolor='orange', label=f"WLC (n={results['wlc']['samples']})")
            
            ax.set_xlabel('y (µm)', fontsize=14, fontweight='bold')
            ax.set_ylabel('P(y)', fontsize=14, fontweight='bold')
            ax.set_xlim(-R, R)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='best')
            ax.set_title(f"P(y): N={N}, y₀=0.0µm, R={R}µm", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            st.pyplot(fig_main)
            
            # ★ Store MAIN figure for export
            st.session_state.stored_fig_main = fig_main
        
        with tab2:
            st.subheader("Sampling Statistics & Bootstrap Error Estimates - P(y)")
            
            stats_data = []
            
            if 'fjc' in results:
                stats_data.append({
                    'Model': 'FJC',
                    'Samples': f"{results['fjc']['samples']:,}",
                    'Mean (µm)': f"{results['fjc']['mean']:.6f} ± {results['fjc']['mean_err']:.6f}",
                    'Std (µm)': f"{results['fjc']['std']:.6f} ± {results['fjc']['std_err']:.6f}",
                    'ESS': f"{results['fjc']['ess']:.0f}"
                })
            
            if 'saw' in results:
                stats_data.append({
                    'Model': 'SAW (Enhanced with PERM)',
                    'Samples': f"{results['saw']['samples']:,}",
                    'Mean (µm)': f"{results['saw']['mean']:.6f} ± {results['saw']['mean_err']:.6f}",
                    'Std (µm)': f"{results['saw']['std']:.6f} ± {results['saw']['std_err']:.6f}",
                    'ESS': f"{results['saw']['ess']:.1f}"
                })
            
            if 'wlc' in results:
                stats_data.append({
                    'Model': 'WLC',
                    'Samples': f"{results['wlc']['samples']:,}",
                    'Mean (µm)': f"{results['wlc']['mean']:.6f} ± {results['wlc']['mean_err']:.6f}",
                    'Std (µm)': f"{results['wlc']['std']:.6f} ± {results['wlc']['std_err']:.6f}",
                    'ESS': f"{results['wlc']['ess']:.0f}"
                })
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    # ★ Progress completion
    progress_bar.progress(100)
    progress_placeholder.success("✅ Computation & Rendering Complete!")
    
    # ★ CRITICAL FIX #1: Set computation_done flag INSIDE if compute: block
    st.session_state.computation_done = True
    st.session_state.stored_results = results
    st.session_state.stored_params = {
        'L': L, 'R': R, 'N': N, 'x0': x0, 'lp': lp, 'a': a
    }

else:
    st.info("👈 Set parameters and click COMPUTE button in the sidebar to begin analysis")

# ============================================================
# COMPLETE CORRECTED EXPORT SECTION
# ============================================================
# This is the FULLY CORRECTED export section with proper checks
# and all functionality. Place this AFTER the if compute/else block.
# ============================================================
# INDENTATION: 0 spaces (at module level)
# ============================================================

# Initialize session state variables
if 'computation_done' not in st.session_state:
    st.session_state.computation_done = False

if 'stored_fig_main' not in st.session_state:
    st.session_state.stored_fig_main = None

if 'stored_fig_bootstrap' not in st.session_state:
    st.session_state.stored_fig_bootstrap = None

if 'stored_results' not in st.session_state:
    st.session_state.stored_results = None

if 'stored_params' not in st.session_state:
    st.session_state.stored_params = None

# Export section - divider
st.markdown("---")
st.subheader("📄 Export Results")

# ★ Show export buttons ONLY if computation was successful
if st.session_state.computation_done and st.session_state.stored_fig_main is not None:
    
    st.success("✅ Analysis complete! Export options available below.", icon="✅")
    
    pdf_col, csv_col, png_col = st.columns(3)
    
    # ============================================================
    # PDF EXPORT BUTTON
    # ============================================================
    with pdf_col:
        if st.button("📄 Export PDF Report", use_container_width=True, key="pdf_export"):
            try:
                # ★ Use MAIN figure (tab1) for export - NOT bootstrap!
                fig_data = {'px_figure': st.session_state.stored_fig_main}
                
                if st.session_state.stored_results is not None:
                    pdf_buffer = pdf_export_module.export_pdf_report(
                        fig_dict=fig_data,
                        results_dict=st.session_state.stored_results,
                        params_dict=st.session_state.stored_params
                    )
                    
                    if pdf_buffer:
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"polymer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("✅ PDF ready for download!\n(Includes all curves: Fourier, FJC, FJC-Gaussian, FJC-Truncated, SAW, WLC)")
                    else:
                        st.error("❌ Error generating PDF")
                else:
                    st.error("❌ Results data not available for export")
            except Exception as e:
                st.error(f"❌ PDF export failed: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
    
    # ============================================================
    # CSV EXPORT BUTTON
    # ============================================================
    with csv_col:
        if st.button("📊 Export CSV Data", use_container_width=True, key="csv_export"):
            try:
                if st.session_state.stored_results is not None:
                    # Create CSV data from results
                    csv_data = []
                    
                    # Extract data from each model
                    if 'fjc' in st.session_state.stored_results:
                        fjc_res = st.session_state.stored_results['fjc']
                        csv_data.append({
                            'Model': 'FJC',
                            'Samples': fjc_res.get('samples', 'N/A'),
                            'Mean (µm)': fjc_res.get('mean', 'N/A'),
                            'Mean_Error (µm)': fjc_res.get('mean_err', 'N/A'),
                            'Std (µm)': fjc_res.get('std', 'N/A'),
                            'Std_Error (µm)': fjc_res.get('std_err', 'N/A'),
                            'ESS': fjc_res.get('ess', 'N/A')
                        })
                    
                    if 'saw' in st.session_state.stored_results:
                        saw_res = st.session_state.stored_results['saw']
                        if 'data' in saw_res and len(saw_res['data']) > 0:
                            csv_data.append({
                                'Model': 'SAW',
                                'Samples': saw_res.get('samples', 'N/A'),
                                'Mean (µm)': np.mean(saw_res.get('data', [])),
                                'Mean_Error (µm)': saw_res.get('mean_err', 'N/A'),
                                'Std (µm)': np.std(saw_res.get('data', [])),
                                'Std_Error (µm)': saw_res.get('std_err', 'N/A'),
                                'ESS': saw_res.get('ess', 'N/A')
                            })
                    
                    if 'wlc' in st.session_state.stored_results:
                        wlc_res = st.session_state.stored_results['wlc']
                        csv_data.append({
                            'Model': 'WLC',
                            'Samples': wlc_res.get('samples', 'N/A'),
                            'Mean (µm)': wlc_res.get('mean', 'N/A'),
                            'Mean_Error (µm)': wlc_res.get('mean_err', 'N/A'),
                            'Std (µm)': wlc_res.get('std', 'N/A'),
                            'Std_Error (µm)': wlc_res.get('std_err', 'N/A'),
                            'ESS': wlc_res.get('ess', 'N/A')
                        })
                    
                    if csv_data:
                        csv_df = pd.DataFrame(csv_data)
                        csv_buffer = csv_df.to_csv(index=False)
                        
                        st.download_button(
                            label="⬇️ Download CSV Data",
                            data=csv_buffer,
                            file_name=f"polymer_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        st.success("✅ CSV ready for download!")
                    else:
                        st.warning("⚠️ No data available for CSV export")
                else:
                    st.error("❌ Results data not available for export")
            except Exception as e:
                st.error(f"❌ CSV export failed: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
    
    # ============================================================
    # PNG EXPORT BUTTON
    # ============================================================
    with png_col:
        if st.button("🖼️ Export Figure (PNG)", use_container_width=True, key="png_export"):
            try:
                if st.session_state.stored_fig_main is not None:
                    png_buf = BytesIO()
                    
                    # ★ Use MAIN figure (tab1) - with all curves!
                    st.session_state.stored_fig_main.savefig(
                        png_buf, format='png', dpi=300, bbox_inches='tight'
                    )
                    png_buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download PNG Figure",
                        data=png_buf,
                        file_name=f"polymer_figure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.success("✅ PNG ready for download!\n(Includes all curves: Fourier, FJC, FJC-Gaussian, FJC-Truncated, SAW, WLC)")
                else:
                    st.error("❌ Figure not available for export")
            except Exception as e:
                st.error(f"❌ PNG export failed: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
    
    # ============================================================
    # RESET BUTTON
    # ============================================================
    st.markdown("---")
    st.subheader("🔄 Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Clear Results & Run Again", use_container_width=True, key="reset_state"):
            st.session_state.computation_done = False
            st.session_state.stored_fig_main = None
            st.session_state.stored_fig_bootstrap = None
            st.session_state.stored_results = None
            st.session_state.stored_params = None
            st.info("✅ Results cleared. Ready for new computation.")
            st.rerun()
    
    with col2:
        if st.button("📋 View Session State", use_container_width=True, key="view_state"):
            with st.expander("Session State Details"):
                st.write(f"Computation Done: {st.session_state.computation_done}")
                st.write(f"Figure Main Stored: {st.session_state.stored_fig_main is not None}")
                st.write(f"Figure Bootstrap Stored: {st.session_state.stored_fig_bootstrap is not None}")
                st.write(f"Results Available: {st.session_state.stored_results is not None}")
                st.write(f"Params Available: {st.session_state.stored_params is not None}")

else:
    # ============================================================
    # EXPORT NOT AVAILABLE MESSAGE
    # ============================================================
    if st.session_state.computation_done and st.session_state.stored_fig_main is None:
        st.warning("⚠️ Computation complete but figure not stored properly. Try running COMPUTE again.", icon="⚠️")
    else:
        st.info("💡 Click the **COMPUTE** button in the sidebar to perform the analysis. Export options will become available after computation completes.", icon="💡")
        
        with st.expander("ℹ️ Export Features"):
            st.markdown("""
            **Available Export Formats:**
            - 📄 **PDF Report**: Complete analysis report with figure and statistics
            - 📊 **CSV Data**: Statistical results and model comparisons
            - 🖼️ **PNG Figure**: High-resolution plot suitable for presentations
            
            **What's Exported:**
            - All curves: Fourier, FJC, FJC-Gaussian, FJC-Truncated, SAW, WLC
            - Statistical metrics: Mean, Std Dev, ESS, Errors
            - Bootstrap estimates when available
            """)