import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from itertools import repeat
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import map_coordinates


def make_pad(phase_2D, whitefield, pad_length = 2000, fill = True):
    # Parameters
    phase_shape = phase_2D.shape
    whitefield_shape = whitefield.shape
    N_padx = phase_shape[0] + 2*pad_length
    N_pady = phase_shape[1] + 2*pad_length
    if fill:
        whitefield[whitefield == 0] = np.mean(whitefield) # change to fill zeros with whatever value you want
    # Pad phase and whitefield
    phase_pad = np.pad(phase_2D, pad_length, mode='constant')
    white_pad = np.pad(whitefield, pad_length, mode='constant')
    
    return phase_pad, white_pad

def get_max_psf(white_pad, phase_pad, defoci, M_defoc):
    P_d = np.sqrt(white_pad)*np.exp(1j*(phase_pad-defoci*M_defoc))
    F_d = np.fft.fft2(P_d)
    F_s_d = np.fft.fftshift(F_d)
    F_I_d = np.abs(F_s_d)**2 # intensity
    max_intensity = np.max(F_I_d)
    return max_intensity

def build_quadratic_term(phase_pad, detector_distance, pxsize, wavelength):
    x = np.arange(- phase_pad.shape[1]//2 * pxsize, phase_pad.shape[1]//2 * pxsize, pxsize)
    y = np.arange(- phase_pad.shape[0]//2 * pxsize, phase_pad.shape[0]//2 * pxsize, pxsize)
    XX, YY = np.meshgrid(x,y)
    defocus_term = (XX**2 + YY**2) / (2 * detector_distance) * 2 * np.pi / wavelength
    return defocus_term

def get_psf(white_pad, phase_pad, defoci, M_defoc):
    P_d = np.sqrt(white_pad)*np.exp(1j*(phase_pad-defoci*M_defoc))
    F_d = np.fft.fft2(P_d)
    F_s_d = np.fft.fftshift(F_d)
    F_I_d = np.abs(F_s_d)**2
    # Normalize by N^2
    F_I_d = F_I_d / (phase_pad.shape[0] * phase_pad.shape[1])
    return F_I_d

def defocus_scan(phase_pad, white_pad, nr_positions, defoci_range, detector_distance, pxsize, wavelength):
    defoci = np.linspace(defoci_range[0], defoci_range[-1], nr_positions)
    defocus_term = build_quadratic_term(phase_pad, detector_distance, pxsize,wavelength)
    num_proc = os.cpu_count()
    with Pool(num_proc-1) as p:
        print('Starting the pool...')
        listing = list(p.starmap(get_psf, zip(repeat(white_pad), repeat(phase_pad), defoci, repeat(defocus_term))))
    psfs = np.array(listing)
    print("Defocus scan completed")
    return psfs


def create_square_mask(h, w, center=None, half_width=None):
    """Create a square mask for an image."""
    if center is None:  # use the middle of the image
        center = (int(h/2), int(w/2))
    if half_width is None:  # use half the smallest dimension
        half_width = min(center[0], center[1], h-center[0], w-center[1])

    y, x = np.ogrid[:h, :w]
    # Square mask where max distance in either x or y is less than half_width
    mask = (np.abs(y - center[0]) <= half_width) & (np.abs(x - center[1]) <= half_width)
    return mask

def rms_polynomial_2d(params, xx, yy, wavefront, mask=None):
    """
    Calculate RMS error after subtracting a polynomial surface.
    
    Parameters:
    - params: [a, b, c, d, e, f] for ax²+by²+cxy+dx+ey+f
    - xx, yy: Meshgrid coordinates
    - wavefront: 2D array of wavefront data
    - mask: Boolean mask defining the region to calculate RMS
    
    Returns:
    - rms: Root mean square error
    """
    a, b, c, d, e, f = params
    # Polynomial surface: ax² + by² + cxy + dx + ey + f
    polynomial = a * xx**2 + b * yy**2 + c * xx * yy + d * xx + e * yy + f
    
    residual = wavefront - polynomial
    
    if mask is not None:
        residual = residual[mask]
    
    # Normalize by removing mean
    mean = np.mean(residual)
    rms = np.sqrt(np.mean((residual - mean)**2))
    return rms

def make_optimal_2d(xx, yy, a, b, c, d, e, f, wavefront):
    """Apply 1st order polynomial correction to wavefront"""
    # Polynomial surface: ax² + by² + cxy + dx + ey + f
    polynomial = a * xx**2 + b * yy**2 + c* xx * yy + d * xx + e * yy + f
    return wavefront - polynomial

def optimize_sub_aperture(wavefront, subaperture_center=None, subaperture_half_width=None, 
                         full_aperture_half_width=None):
    """
    Optimizes a square sub-aperture of a wavefront and applies the correction to the full wavefront.
    
    Parameters:
    - wavefront: 2D array of wavefront data
    - subaperture_center: (y, x) coordinates of subaperture center, defaults to center of array
    - subaperture_half_width: Half-width of subaperture, defaults to 1/3 of full aperture
    - full_aperture_half_width: Half-width of full aperture, defaults to half of smallest dimension
    
    Returns:
    - corrected_wavefront: Full wavefront with polynomial terms optimized for subaperture
    - subaperture_mask: Mask showing the optimized region
    - full_mask: Mask showing the full aperture
    - rms_before: RMS of subaperture before correction
    - rms_after: RMS of subaperture after correction
    - params: Optimized polynomial parameters [a, b, c, d, e, f]
    """
    h, w = wavefront.shape
    
    # Set defaults for center and half-widths if not provided
    if subaperture_center is None:
        subaperture_center = (h//2, w//2)
    
    if full_aperture_half_width is None:
        full_aperture_half_width = min(h, w) // 4  # Quarter of smallest dimension
    
    if subaperture_half_width is None:
        subaperture_half_width = full_aperture_half_width // 3
    
    # Create masks for subaperture and full aperture
    subaperture_mask = create_square_mask(h, w, subaperture_center, subaperture_half_width)
    full_mask = create_square_mask(h, w, subaperture_center, full_aperture_half_width)
    
    # Create coordinate meshgrid
    y, x = np.mgrid[:h, :w]
    
    # Center and normalize coordinates for better numerical stability
    x_center = x - subaperture_center[1]
    y_center = y - subaperture_center[0]
    
    # Scale factor for normalization
    scale = max(h, w) / 2
    xx = x_center / scale
    yy = y_center / scale
    
    # Calculate initial RMS in subaperture
    subaperture_data = wavefront[subaperture_mask]
    rms_before = np.sqrt(np.mean((subaperture_data - np.mean(subaperture_data))**2))
    
    # Initial guess for polynomial parameters [a, b, c, d, e, f]
    # For ax² + by² + cxy + dx + ey + f
    initial_params = [0, 0, 0, 0, 0, 0]
    
    # Optimize parameters to minimize RMS in subaperture
    result = minimize(
        rms_polynomial_2d, 
        initial_params, 
        args=(xx, yy, wavefront, subaperture_mask),
        method='Powell',
        tol=1e-6
    )
    
    optimized_params = result.x
    
    # Apply optimized polynomial correction to the full wavefront
    corrected_wavefront = make_optimal_2d(
        xx, yy, 
        optimized_params[0], optimized_params[1], optimized_params[2],
        optimized_params[3], optimized_params[4], optimized_params[5],
        wavefront
    )
    
    # Normalize by removing mean from the corrected wavefront
    corrected_wavefront = corrected_wavefront - np.mean(corrected_wavefront[full_mask])
    
    # Calculate final RMS in subaperture
    corrected_subaperture = corrected_wavefront[subaperture_mask]
    rms_after = np.sqrt(np.mean((corrected_subaperture - np.mean(corrected_subaperture))**2))
    
    return (corrected_wavefront, subaperture_mask, full_mask, 
            rms_before, rms_after, optimized_params)

def subtract_y_tilt(wavefront, plot = False):
    ny, nx = wavefront.shape
    y = np.arange(ny)
    x = np.arange(nx)
    Y, X = np.meshgrid(y, x, indexing='ij')  # Shape (ny, nx)

    # Flatten the arrays to fit a plane: W = a*x + b*y + c
    A = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))
    coeffs, _, _, _ = np.linalg.lstsq(A, wavefront.ravel(), rcond=None)
    a, b, c = coeffs

    # Subtract only the y-tilt component: b * Y
    y_tilt = b * Y
    if plot:
        plt.figure()
        plt.imshow(y_tilt)
        plt.colorbar()
        plt.title("Subtracted Y tilt")
        plt.show()
    corrected_wavefront = wavefront - y_tilt
    corrected_wavefront -= corrected_wavefront.mean()

    return corrected_wavefront

def subtract_x_tilt(wavefront, plot = False):
    ny, nx = wavefront.shape
    y = np.arange(ny)
    x = np.arange(nx)
    Y, X = np.meshgrid(y, x, indexing='ij')  # Shape (ny, nx)

    # Flatten the arrays to fit a plane: W = a*x + b*y + c
    A = np.column_stack((X.ravel(), Y.ravel(), np.ones_like(X.ravel())))
    coeffs, _, _, _ = np.linalg.lstsq(A, wavefront.ravel(), rcond=None)
    a, b, c = coeffs

    # Subtract only the y-tilt component: b * Y
    x_tilt = a * X
    if plot:
        plt.figure()
        plt.imshow(x_tilt)
        plt.colorbar()
        plt.title("Subtracted X tilt")
        plt.show()
    corrected_wavefront = wavefront - x_tilt
    corrected_wavefront -= corrected_wavefront.mean()
    
    return corrected_wavefront

def subtract_linear_term(wavefront, plot = False):
    corr_x = subtract_x_tilt(wavefront, plot = plot)
    corr_2d = subtract_y_tilt(corr_x, plot = plot)
    return corr_2d

def get_focus_values(PSF, full_shape, pxsize, wavelength, distance, plot = True, return_cuts = False):
    brightest_point_index = np.argmax(PSF)
    brightest_y, brightest_x = np.unravel_index(brightest_point_index, PSF.shape)
    # print(f"Brightest point: ({brightest_x}, {brightest_y})")

    # Calculate maximum safe distance for diagonal cuts to stay within image bounds
    max_length = min(brightest_y, brightest_x, 
                    PSF.shape[0] - brightest_y - 1, 
                    PSF.shape[1] - brightest_x - 1)

    # Generate coordinate arrays for diagonal cuts
    coords_range = np.arange(-max_length, max_length + 1)

    # 45° diagonal (bottom-left to top-right): x and y increments are identical
    diagonal_y_45 = brightest_y + coords_range
    diagonal_x_45 = brightest_x + coords_range

    # 135° diagonal (top-left to bottom-right): x and y increments are opposite
    diagonal_y_135 = brightest_y + coords_range
    diagonal_x_135 = brightest_x - coords_range

    # Extract all cuts
    cuts = {
        'vertical': PSF[:, brightest_x],
        'horizontal': PSF[brightest_y, :],
        'diagonal_45': map_coordinates(PSF, [diagonal_y_45, diagonal_x_45], order=1),
        'diagonal_135': map_coordinates(PSF, [diagonal_y_135, diagonal_x_135], order=1)
    }
    if plot:
        fig, axs = plt.subplots(1,2, figsize=(11, 4), constrained_layout = True)

        im = axs[0].imshow(PSF, cmap = "gist_heat")
        axs[0].plot(diagonal_x_45, diagonal_y_45, color="magenta", linewidth=1, alpha=.7, label="45° diagonal")
        axs[0].plot(diagonal_x_135, diagonal_y_135, color="green", linewidth=1, alpha=.7,label="135° diagonal")
        axs[0].axhline(y=brightest_y, color="blue", linewidth=1,alpha=.7, label="Horizontal cut")
        axs[0].axvline(x=brightest_x, color="orange", linewidth=1,alpha=.7, label="Vertical cut")
        axs[0].scatter(brightest_x, brightest_y, color="red", s=50, zorder=5, label="Brightest point")
        plt.colorbar(im, ax=axs[0])
        axs[0].set_title('Cut Locations on 2D Image')
        # axs[0].legend()

        colors = ['orange', 'blue', 'magenta', 'green']
        for (label, profile_data), color in zip(cuts.items(), colors):
            axs[1].plot(profile_data, label=label, color=color)

        axs[1].legend()
        axs[1].set_title('1D Focus Profiles')
        axs[1].set_xlabel('Position')
        axs[1].set_ylabel('Intensity')
        axs[1].grid(True, alpha=0.3)

        # plt.savefig("results/psf/scan_36/1D_profiles.png", dpi = 150, bbox_inches = "tight")
        plt.show()

    # Gaussian fitting for 1D focus profiles
    def gaus(x, a, x0, sigma):
        """Gaussian function for curve fitting."""
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Prepare data for fitting
    cut_names = ['vertical', 'horizontal', 'diagonal_45', 'diagonal_135']
    cut_data = [cuts[name] for name in cut_names]

    # Initial guess parameters for each cut [amplitude, center, sigma]
    initial_params = [
        [1, brightest_y, 20],  # vertical cut - center at brightest_y
        [1, brightest_x, 20],  # horizontal cut - center at brightest_x
        [1, len(cuts['diagonal_45'])//2, 20],  # diagonal_45 - center at middle
        [1, len(cuts['diagonal_135'])//2, 20]  # diagonal_135 - center at middle
    ]

    # Perform fitting for all cuts
    fitted_params = []
    fitted_curves = []

    for i, (cut, init_params) in enumerate(zip(cut_data, initial_params)):
        # Normalize cut data
        cut_normalized = cut / np.max(cut)
        x_coords = np.arange(len(cut))
        
        # Fit Gaussian
        try:
            popt, pcov = curve_fit(gaus, x_coords, cut_normalized, 
                                p0=init_params, maxfev=10000)
            fitted_curve = gaus(x_coords, *popt)
            
            fitted_params.append(popt)
            fitted_curves.append((x_coords, cut_normalized, fitted_curve))
            
        except Exception as e:
            print(f"Fitting failed for {cut_names[i]}: {e}")
            fitted_params.append([np.nan, np.nan, np.nan])
            fitted_curves.append((x_coords, cut_normalized, np.full_like(x_coords, np.nan)))

    # Create visualization
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()

        titles = ['Vertical Cut', 'Horizontal Cut', '45° Diagonal Cut', '135° Diagonal Cut']
        colors = ['orange', 'blue', 'yellow', 'green']

        for i, (ax, title, color) in enumerate(zip(axs, titles, colors)):
            if i < len(fitted_curves):
                x_coords, cut_normalized, fitted_curve = fitted_curves[i]
                
                ax.plot(x_coords, cut_normalized, 'o-', color=color, alpha=0.7, 
                        markersize=3, label='Data')
                ax.plot(x_coords, fitted_curve, '--', color='red', linewidth=2, 
                        label='Gaussian fit')
                
                # Add fit parameters as text
                if not np.isnan(fitted_params[i][0]):
                    params_text = f'σ = {fitted_params[i][2]:.2f}'
                    ax.text(0.05, 0.95, params_text, transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(title)
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Normalized Intensity')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.savefig("results/psf/scan_36/1D_profiles_fits.png", dpi=100, bbox_inches = "tight")
        plt.show()

    # Print fitting results
    print("\nGaussian Fitting Results:")
    print("Cut Name          | Amplitude | Center   | Sigma")
    print("-" * 50)
    for i, (name, params) in enumerate(zip(cut_names, fitted_params)):
        if not np.isnan(params[0]):
            print(f"{name:15s} | {params[0]:8.3f} | {params[1]:7.2f} | {params[2]:6.2f}")
        else:
            print(f"{name:15s} | Failed to fit")

    # Calculate and display FWHM (Full Width at Half Maximum)
    print("\nFWHM (Full Width at Half Maximum):")
    for i, (name, params) in enumerate(zip(cut_names, fitted_params)):
        if not np.isnan(params[2]):
            fwhm = 2.355 * params[2]  # FWHM = 2.355 * sigma for Gaussian
            print(f"{name:15s}: {fwhm:.2f} pixels")
        else:
            print(f"{name:15s}: N/A")

    N_padx, N_pady = full_shape
    dpx_horz = 1 / (N_padx * pxsize / (wavelength * distance))
    dpx_vert = 1 / (N_pady * pxsize / (wavelength * distance))

    print(f'Horizontal pixel size: {dpx_horz:.2e} m')
    print(f'Vertical pixel size: {dpx_vert:.2e} m')

    # Conversion factor from sigma to FWHM for Gaussian
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    # Calculate physical focus sizes for each cut
    cut_pixel_sizes = [dpx_vert, dpx_horz, dpx_horz, dpx_vert]  # pixel sizes for each cut direction
    cut_labels = ['Vertical', 'Horizontal', '45° Diagonal', '135° Diagonal']

    print("\nPhysical Focus Sizes:")
    print("Direction        | Sigma (pixels) | FWHM (pixels) | Physical Size (nm)")
    print("-" * 70)

    physical_sizes = []
    for i, (label, pixel_size, params) in enumerate(zip(cut_labels, cut_pixel_sizes, fitted_params)):
        if not np.isnan(params[2]):
            sigma_pixels = np.abs(params[2])
            fwhm_pixels = fwhm_factor * sigma_pixels
            physical_size = pixel_size * fwhm_pixels  # Physical size in meters
            physical_size_nm = physical_size * 1e9    # Convert to nanometers
            
            physical_sizes.append(physical_size_nm)
            print(f"{label:15s} | {sigma_pixels:13.2f} | {fwhm_pixels:12.2f} | {physical_size_nm:15.1f}")
        else:
            physical_sizes.append(np.nan)
            print(f"{label:15s} | {'N/A':13s} | {'N/A':12s} | {'N/A':15s}")

    # Summary statistics
    valid_sizes = [size for size in physical_sizes if not np.isnan(size)]
    if valid_sizes:
        print(f"\nSummary:")
        print(f"Average focus size: {np.mean(valid_sizes):.1f} nm")
        print(f"Size variation (std): {np.std(valid_sizes):.1f} nm")
        print(f"Aspect ratio (H/V): {physical_sizes[1]/physical_sizes[0]:.2f}" if not np.isnan(physical_sizes[0]) and not np.isnan(physical_sizes[1]) else "N/A")
    
    out = dict(zip(cut_labels, valid_sizes))
    out["virt_pxsize_hor"] = dpx_horz*1e9
    out["virt_pxsize_vert"] = dpx_vert*1e9
    if return_cuts:
        return cuts, out
    else: 
        return out



# Function to calculate distance grid centered around the brightest point
def get_distance_grid(shape, center_x, center_y):
    height, width = shape
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return distance_from_center

# Function to calculate encircled energy with the brightest point as center
def get_encircled_energy(image, pix_size, coords, nr_points=200):
    shape = image.shape
    brightest_x, brightest_y = coords

    max_rad = min(brightest_x, brightest_y, shape[1] - brightest_x, shape[0] - brightest_y)
    radii = np.linspace(1, max_rad, nr_points)

    # Precompute the distance grid centered around the brightest point
    distance_grid = get_distance_grid(shape, brightest_x, brightest_y)

    # Flatten the image and distance grid for efficient processing
    flattened_image = image.flatten()
    flattened_distance_grid = distance_grid.flatten()

    # Sort the image pixels by their distance from the brightest point
    sorted_indices = np.argsort(flattened_distance_grid)
    sorted_distances = flattened_distance_grid[sorted_indices]
    sorted_image = flattened_image[sorted_indices]

    # Compute cumulative sum of the sorted image values
    cumulative_sum = np.cumsum(sorted_image)

    # Calculate encircled energy for each radius
    encircled_energy = []
    for radius in radii:
        # Find the maximum index within the current radius
        max_index = np.searchsorted(sorted_distances, radius, side='right')
        encircled_energy.append(cumulative_sum[max_index-1] if max_index > 0 else 0)

    return radii, encircled_energy

# Wrapper function for multiprocessing
def process_single_psf(args):
    psf, pix_size, coords, nr_points = args
    psf_norm = psf / np.sum(psf)
    return get_encircled_energy(psf_norm, pix_size, coords, nr_points)[1]  # Return only encircled energy

# Main processing function with parallelization
def calculate_encircled_energy_parallel(psfs, pix_size, coords, nr_points=200):
    num_cores = min(cpu_count(), len(psfs))  # Limit cores to number of PSFs
    args = [(psf, pix_size, coords, nr_points) for psf in psfs]

    with Pool(num_cores) as pool:
        encircled_energy_list = list(pool.imap(process_single_psf, args))

    return np.array(encircled_energy_list)

