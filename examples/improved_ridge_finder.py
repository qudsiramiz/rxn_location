import numpy as np
from scipy.optimize import minimize_scalar

def find_xline_radial_bisection(eval_func, center_yz=(0, 0), theta_steps=180, r_bounds=(0.1, 20.0)):
    # Old radial method left for reference
    y_ridge = []
    z_ridge = []
    thetas = np.linspace(0, 2 * np.pi, theta_steps, endpoint=False)
    for theta in thetas:
        def objective_slice(R):
            y_loc = center_yz[0] + R * np.cos(theta)
            z_loc = center_yz[1] + R * np.sin(theta)
            return -eval_func(y_loc, z_loc)
        res = minimize_scalar(objective_slice, bounds=r_bounds, method='bounded')
        if res.success:
            best_R = res.x
            y_ridge.append(center_yz[0] + best_R * np.cos(theta))
            z_ridge.append(center_yz[1] + best_R * np.sin(theta))
    return np.array(y_ridge), np.array(z_ridge)


def trace_bisection_xline(y_start, z_start, get_b_msh, get_b_msp, step_size=0.25, max_steps=500, bounds=20, enforce_monotonic_y=True):
    """
    Traces the continuous reconnection X-line by integrating along the 
    magnetic bisector vector field, as described in Qudsi et al. (2023).
    
    Parameters
    ----------
    y_start, z_start : float
        The Y and Z GSM coordinates of the maximum reconnecting field 
        (the starting point for the integration).
    get_b_msh, get_b_msp : callable
        Functions f(y, z) that return the 3D local magnetic field vector (np.array([Bx, By, Bz]))
        for the magnetosheath and magnetosphere, respectively.
    step_size : float
        The integration step size in Earth Radii (R_E).
    max_steps : int
        Maximum number of steps to integrate in each direction.
    bounds : float
        The maximum absolute value for y and z before stopping the trace.
    enforce_monotonic_y : bool
        If True, the trace will strictly move in the same Y-direction, avoiding looping back on itself.
        
    Returns
    -------
    y_line, z_line : numpy.ndarray
        The Y and Z coordinates of the smooth, continuous X-line.
    """
    
    def get_bisector_direction(y, z):
        b_msh = get_b_msh(y, z)
        b_msp = get_b_msp(y, z)
        
        if b_msh is None or b_msp is None or np.any(np.isnan(b_msh)) or np.any(np.isnan(b_msp)):
            return np.array([0.0, 0.0])
            
        mag_msh = np.linalg.norm(b_msh)
        mag_msp = np.linalg.norm(b_msp)
        
        if mag_msh == 0 or mag_msp == 0:
            return np.array([0.0, 0.0])
            
        unit_msh = b_msh / mag_msh
        unit_msp = b_msp / mag_msp
        
        # The bisector defines the direction of the X-line in 3D
        bisector_3d = unit_msh + unit_msp
        b_mag = np.linalg.norm(bisector_3d)
        if b_mag > 0:
            bisector_3d /= b_mag
            
        # Project the 3D bisector onto the 2D YZ plane for tracing
        direction_2d = np.array([bisector_3d[1], bisector_3d[2]])
        
        # Normalize the 2D step direction
        dir_mag = np.linalg.norm(direction_2d)
        if dir_mag > 0:
            direction_2d /= dir_mag
            
        return direction_2d

    y_line_pos, z_line_pos = [y_start], [z_start]
    y_line_neg, z_line_neg = [], []
    
    # ---------------------------------------------------------
    # Integrate Forward (Positive direction along the bisector)
    # ---------------------------------------------------------
    curr_y, curr_z = y_start, z_start
    prev_dir = None
    initial_y_sign = None
    for step in range(max_steps):
        direction = get_bisector_direction(curr_y, curr_z)
        if np.all(direction == 0):
            break
            
        if step == 0:
            # Establish the forward Y direction
            initial_y_sign = np.sign(direction[0])
            if initial_y_sign == 0:
                initial_y_sign = 1
                
        if enforce_monotonic_y:
            # If the direction flips Y sign, it means the X-line folds back on itself.
            # We break instead of flipping the vector to avoid zig-zagging in Z.
            if np.sign(direction[0]) != initial_y_sign and direction[0] != 0:
                break
        else:
            # Fallback to dot-product method
            if prev_dir is not None:
                if np.dot(direction, prev_dir) < 0:
                    direction = -direction
                    
        prev_dir = direction
            
        curr_y += step_size * direction[0]
        curr_z += step_size * direction[1]
        
        if abs(curr_y) > bounds or abs(curr_z) > bounds: 
            break
            
        y_line_pos.append(curr_y)
        z_line_pos.append(curr_z)

    # ---------------------------------------------------------
    # Integrate Backward (Negative direction along the bisector)
    # ---------------------------------------------------------
    curr_y, curr_z = y_start, z_start
    prev_dir = None
    initial_y_sign_backward = None
    for step in range(max_steps):
        direction = get_bisector_direction(curr_y, curr_z)
        if np.all(direction == 0):
            break
            
        if step == 0:
            # Establish the backward Y direction (opposite of forward)
            initial_y_sign_backward = -initial_y_sign
            
        if enforce_monotonic_y:
            # If the direction flips Y sign, it means the X-line folds back on itself.
            if np.sign(direction[0]) != initial_y_sign_backward and direction[0] != 0:
                break
        else:
            # For the very first step, force it to go backwards
            if step == 0:
                direction = -direction
            if prev_dir is not None:
                if np.dot(direction, prev_dir) < 0:
                    direction = -direction
                    
        prev_dir = direction
            
        curr_y += step_size * direction[0]
        curr_z += step_size * direction[1]
        
        if abs(curr_y) > bounds or abs(curr_z) > bounds:
            break
            
        y_line_neg.append(curr_y)
        z_line_neg.append(curr_z)

    # Combine the backward and forward traces into one continuous line
    y_line = np.array(y_line_neg[::-1] + y_line_pos)
    z_line = np.array(z_line_neg[::-1] + z_line_pos)
    
    return y_line, z_line
