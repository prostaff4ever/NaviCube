
import numpy as np
from numpy.fft import fftn, fftshift, ifftn
from scipy.ndimage import gaussian_filter

# --- PARAMETERS ---
N = 32
step = 4
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
z = np.linspace(-1, 1, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- FUNCTION DEFINITIONS ---
def magnetic_dipole_field(x, y, z, x0, y0, z0, m):
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2) + 1e-6
    r_hat_x = (x - x0) / r
    r_hat_y = (y - y0) / r
    r_hat_z = (z - z0) / r
    dot = m * r_hat_z
    Bx = (3 * r_hat_x * dot) / (r**3)
    By = (3 * r_hat_y * dot) / (r**3)
    Bz = (3 * r_hat_z * dot - m) / (r**3)
    return Bx, By, Bz

def generate_1f_noise(shape, amplitude_scale=1.0):
    kx = np.fft.fftfreq(shape[0]).reshape(-1, 1, 1)
    ky = np.fft.fftfreq(shape[1]).reshape(1, -1, 1)
    kz = np.fft.fftfreq(shape[2]).reshape(1, 1, -1)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0, 0, 0] = 1e-6
    spectrum = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    spectrum *= (1.0 / k_mag)
    noise = np.real(ifftn(spectrum))
    noise -= np.mean(noise)
    noise /= np.max(np.abs(noise))
    return noise * amplitude_scale

def fft_mag(Bx, By, Bz):
    return fftshift(np.abs(fftn(np.sqrt(Bx**2 + By**2 + Bz**2))))

# --- DIPOLES & BASELINE SETUP ---
dipoles = [
    {"pos": (-2.5, 0.7, -1.2), "m": 1.0},
    {"pos": (-0.5, -0.5, -0.5), "m": 0.3},
    {"pos": (0.5, 0.5, 0.5), "m": 0.5},
    {"pos": (-0.7, 0.6, 0.0), "m": 0.2}
]
baseline_vector = (-0.3, -0.5, -0.7)

# --- FIELD GENERATION ---
Bx, By, Bz = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
for d in dipoles:
    bx, by, bz = magnetic_dipole_field(X, Y, Z, *d["pos"], d["m"])
    Bx += bx
    By += by
    Bz += bz
Bx += baseline_vector[0]
By += baseline_vector[1]
Bz += baseline_vector[2]

# --- NOISE GENERATION ---
smallest_m = min(d["m"] for d in dipoles)
np.random.seed(42)
gauss_1p = 0.01 * smallest_m
gauss_10p = 0.10 * smallest_m
pink_5p = 0.05 * smallest_m

Bx_1p = Bx + np.random.normal(0, gauss_1p, Bx.shape)
By_1p = By + np.random.normal(0, gauss_1p, By.shape)
Bz_1p = Bz + np.random.normal(0, gauss_1p, Bz.shape)

Bx_10p = Bx + np.random.normal(0, gauss_10p, Bx.shape)
By_10p = By + np.random.normal(0, gauss_10p, By.shape)
Bz_10p = Bz + np.random.normal(0, gauss_10p, Bz.shape)

pink_noise = generate_1f_noise(Bx.shape, pink_5p)
Bx_combined = Bx_10p + pink_noise
By_combined = By_10p + pink_noise
Bz_combined = Bz_10p + pink_noise

# --- FILTERING ---
sigma = 1.0
Bx_filt = gaussian_filter(Bx_1p, sigma=sigma)
By_filt = gaussian_filter(By_1p, sigma=sigma)
Bz_filt = gaussian_filter(Bz_1p, sigma=sigma)

# --- UNDERSAMPLING FOR ALIASING ---
factor = 2
Bx_alias = Bx_1p[::factor, ::factor, ::factor]
By_alias = By_1p[::factor, ::factor, ::factor]
Bz_alias = Bz_1p[::factor, ::factor, ::factor]

Bx_lp_alias = Bx_filt[::factor, ::factor, ::factor]
By_lp_alias = By_filt[::factor, ::factor, ::factor]
Bz_lp_alias = Bz_filt[::factor, ::factor, ::factor]

# --- MAGNITUDES & FFTs ---
fft_clean = fft_mag(Bx, By, Bz)
fft_1p = fft_mag(Bx_1p, By_1p, Bz_1p)
fft_10p = fft_mag(Bx_10p, By_10p, Bz_10p)
fft_combined = fft_mag(Bx_combined, By_combined, Bz_combined)
fft_alias = fft_mag(Bx_alias, By_alias, Bz_alias)
fft_lp_alias = fft_mag(Bx_lp_alias, By_lp_alias, Bz_lp_alias)
