
import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from scipy.optimize import least_squares
from skimage.feature import peak_local_max
from numpy.fft import fftn, fftshift, ifftn

# ------------------------------
# MODULE 1: Localization
# ------------------------------

def localize_dipoles_by_gradient(Bx, By, Bz, percentile_threshold=95, min_distance=2):
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    dBx = sobel(B_mag, axis=0)
    dBy = sobel(B_mag, axis=1)
    dBz = sobel(B_mag, axis=2)
    grad_magnitude = np.sqrt(dBx**2 + dBy**2 + dBz**2)
    threshold = np.percentile(grad_magnitude, percentile_threshold)
    peaks = peak_local_max(grad_magnitude, min_distance=min_distance,
                           threshold_abs=threshold, exclude_border=False)
    peak_coords = [tuple(coord) for coord in peaks]
    return peak_coords, grad_magnitude

# ------------------------------
# MODULE 2: Parametric Fitting
# ------------------------------

def dipole_field_model(grid, position, moment):
    x, y, z = grid
    x0, y0, z0 = position
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2) + 1e-6
    r_hat_x = (x - x0) / r
    r_hat_y = (y - y0) / r
    r_hat_z = (z - z0) / r
    dot = moment * r_hat_z
    Bx = (3 * r_hat_x * dot) / (r**3)
    By = (3 * r_hat_y * dot) / (r**3)
    Bz = (3 * r_hat_z * dot - moment) / (r**3)
    return Bx, By, Bz

def flatten_field(Bx, By, Bz):
    return np.concatenate([Bx.ravel(), By.ravel(), Bz.ravel()])

def residuals(params, grid, Bx_meas, By_meas, Bz_meas):
    x0, y0, z0, m = params
    Bx_model, By_model, Bz_model = dipole_field_model(grid, (x0, y0, z0), m)
    return flatten_field(Bx_model - Bx_meas, By_model - By_meas, Bz_model - Bz_meas)

def fit_dipole(grid, Bx_meas, By_meas, Bz_meas, init_position, init_moment=1.0):
    init_params = [*init_position, init_moment]
    result = least_squares(
        residuals, init_params, args=(grid, Bx_meas, By_meas, Bz_meas),
        bounds=([-np.inf]*3 + [0], [np.inf]*3 + [np.inf]), method='trf'
    )
    x0, y0, z0, m = result.x
    return (x0, y0, z0), m, result

# ------------------------------
# MODULE 3: Dipole Subtraction
# ------------------------------

def subtract_dipoles(Bx_meas, By_meas, Bz_meas, grid, dipole_list):
    X, Y, Z = grid
    Bx_dipoles = np.zeros_like(Bx_meas)
    By_dipoles = np.zeros_like(By_meas)
    Bz_dipoles = np.zeros_like(Bz_meas)

    for (pos, moment) in dipole_list:
        bx, by, bz = dipole_field_model(grid, pos, moment)
        Bx_dipoles += bx
        By_dipoles += by
        Bz_dipoles += bz

    Bx_resid = Bx_meas - Bx_dipoles
    By_resid = By_meas - By_dipoles
    Bz_resid = Bz_meas - Bz_dipoles

    return Bx_resid, By_resid, Bz_resid, Bx_dipoles, By_dipoles, Bz_dipoles
