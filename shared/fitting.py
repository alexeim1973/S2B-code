from photutils.isophote import Ellipse, EllipseGeometry
import Fourier_functions as mf
from Fourier_params import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import pynbody
import numpy as np
from scipy import stats as st
from scipy.signal import find_peaks


model = 'TNG50'  # TNG50 skinny bar from Stuart Andersson
base_dir = '/home/ubuntu/projects/S2B/'
model_dir = base_dir + "models/" + model + "/"
image_dir = base_dir + "images/bar-shapes/" + model + "/"
fname = "455291_99"
dens_image_fname = fname + "-dens-ellipse-fit-br.png"
ellips_image_fname = fname + "-ellips-ellipse-fit-br.png"
file = model_dir + fname
paramfile = model_dir + model + '.param'
bins = 30
xlim, ylim = 7, 7
num_ellipses = 30


def load_snapshot_and_density(fname, paramfile, bins, xlim, ylim, verbose=True):
    sn = mf.pbload(fname, paramfile, ncl_bar, bar_align, log)
    sn.physical_units()

    if verbose:
        print("Snapshot is centered and aligned face-on.")

    # Extract NP arrays and mask central area xlim by ylim
    x,y,z,m,age,tf = mf.extract_np(sn)
    x,y,z,m,age,tf = mf.mask_np(x,y,z,m,age,tf,xlim,ylim)

    # Number density statistics face-on for stellar population by age group
    df_stat2d,df_xedges,df_yedges,df_binnum2d = st.binned_statistic_2d(x, y, z,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
    
    #density, _, _ = np.histogram2d(x, y, bins=bins, range=[[-xlim, xlim], [-ylim, ylim]])
    density = df_stat2d.T

    density[np.isinf(density)] = 0
    density[np.isnan(density)] = 0

    if verbose:
        print(f"Shape: {density.shape}, X range: {x.min():.1f}–{x.max():.1f}, Y range: {y.min():.1f}–{y.max():.1f}")

    return sn, density


def get_density_peak_center(density, verbose=True):
    y_peak, x_peak = np.unravel_index(np.argmax(density), density.shape)
    if verbose:
        print("Density peak center is:", x_peak, y_peak)
    return x_peak, y_peak


def fit_ellipses_to_density(density, bins, num_ellipses, verbose=True):

    #xs, ys = get_density_peak_center(density)
    #xs, ys = 11, 11

    ys, xs = np.unravel_index(np.argmax(density), density.shape)
    if verbose:
        print("Density peak center is:", xs, ys)

    sma = bins / 4
    eps = 0.2
    pa = 1. / 180. * np.pi

    g = EllipseGeometry(xs, ys, sma, eps, pa)
    g.fix_center = True
    ellipse = Ellipse(density, geometry=g)
    estep = bins / (num_ellipses - 1) / 2
    isolist = ellipse.fit_image(step=estep, maxsma=0.95*bins/2, linear=True, maxgerr=2)

    if verbose:
        center_pix = (density.shape[1] // 2, density.shape[0] // 2)
        print("Density array center (x, y):", center_pix)
        t = isolist.to_table()
        print(t.colnames)
        print("Isophotes center, ellipticity, semi-major axis, position angle")
        print(t['x0', 'y0', 'ellipticity', 'sma', 'pa'][:20])  # First few isophotes

    return isolist


def plot_density_with_ellipses_br(
    density,
    isolist,
    xlim,
    ylim,
    bins,
    image_path=None,
    verbose=True,
    show_bar_radius=True
):
    t = isolist.to_table()

    smas = t['sma']
    epsilons = t['ellipticity']
    pas_deg = t['pa']  # position angle in degrees

    fig, ax1 = plt.subplots(figsize=(6, 6))

    # Density map
    plt.imshow(
        density,
        origin='lower',
        extent=[-xlim, xlim, -ylim, ylim],
        norm=LogNorm(),
        cmap=cmap_dens
    )

    # Plot fitted ellipses
    for sma in smas:
        iso = isolist.get_closest(sma)
        x1, y1 = iso.sampled_coordinates()
        x2 = 0.5 + x1 - bins / 2
        y2 = 0.5 + y1 - bins / 2
        x2 *= (2 * xlim) / bins
        y2 *= (2 * ylim) / bins
        ax1.plot(x2, y2, color='white', lw=1, solid_joinstyle='round')

    # Mark centers
    ax1.plot(0, 0, 'rx', markersize=10, label='Density Center')
    x0_mean = np.mean(t['x0'])
    y0_mean = np.mean(t['y0'])
    x0_phys = (0.25 + x0_mean - bins/2) * (2 * xlim / bins)
    y0_phys = (0.5 + y0_mean - bins/2) * (2 * ylim / bins)
    ax1.plot(x0_phys, y0_phys, 'b+', markersize=10, label='Ellipse Fit Center')

    # --- BAR LENGTH ESTIMATE 1: Peak Ellipticity ---
    if show_bar_radius and len(smas) > 0:
        idx_max_e = np.argmax(epsilons)
        sma_peak_e_kpc = smas[idx_max_e] * (2 * xlim / bins)
        bar_circle_e = Circle((0, 0), sma_peak_e_kpc, color='yellow', ls='--', lw=2, fill=False,
                              label='Bar Radius (peak e)')
        ax1.add_patch(bar_circle_e)
        if verbose:
            print(f"[INFO] Bar radius (peak ellipticity): {sma_peak_e_kpc:.2f} kpc")

    # --- BAR LENGTH ESTIMATE 2: PA deviation < +-1 deg from horisontal line ---
    if show_bar_radius and len(pas_deg) > 0:
        deviation_threshold_deg = 1

        # Fix: handle Astropy Quantity with units
        if hasattr(pas_deg, 'value'):
            pa_deviation = np.minimum(np.abs(pas_deg.value), np.abs(180 - pas_deg.value))
        else:
            pa_deviation = np.minimum(np.abs(pas_deg), np.abs(180 - pas_deg))

        within_threshold = pa_deviation < deviation_threshold_deg

        # Now find the last isophote where PA is still aligned (i.e., within threshold)
        if np.any(within_threshold):
            last_good_idx = np.where(within_threshold)[0][-1]
            bar_radius_pa = smas[last_good_idx] * (2 * xlim / bins)
            print(f"[INFO] Bar radius (PA aligned within ±{deviation_threshold_deg}°): {bar_radius_pa:.2f} kpc")
        else:
            bar_radius_pa = None
            print("[WARNING] No PA-aligned isophotes found within threshold.")

        if bar_radius_pa is not None:
            bar_circle_pa = Circle((0, 0), bar_radius_pa, color='cyan', ls=':', lw=2, fill=False,
                                   label='Bar Radius (PA dev < 1°)')
            ax1.add_patch(bar_circle_pa)

    # Final plot settings
    ax1.set_title("Isophote Fitting over Density")
    ax1.set_xlabel("X [kpc]")
    ax1.set_ylabel("Y [kpc]")
    ax1.legend()

    if image_path:
        plt.savefig(image_path)
        print("Image saved to", image_path)
        plt.close()
    else:
        plt.show()


def plot_epsilon_vs_sma(
    isolist,
    image_path=None,
    verbose=True
):
    t = isolist.to_table()

    smas = t['sma']
    epsilons = t['ellipticity']
    smas_kpc = smas * (2 * xlim / bins) # smas in kpc
    pas_deg = t['pa']  # position angle in degrees

    # Plotting ellipticity vs semi-major axis
    plt.figure(figsize=(8, 5))
    plt.plot(smas_kpc, epsilons, marker='o', linestyle='-')
    plt.xlabel('Semi-Major Axis (sma) [kpc]')
    plt.ylabel('Ellipticity (ε)')
    plt.title('Ellipticity vs Semi-Major Axis')
    plt.grid(True)

    # Highlight peak ellipticity
    idx_peak = np.argmax(epsilons)
    plt.axvline(smas_kpc[idx_peak], color='red', linestyle='--', label=f'Peak ε = {epsilons[idx_peak]:.3f} at sma = {smas_kpc[idx_peak]:.2f}')
    plt.legend()

    plt.tight_layout()

    if image_path:
        plt.savefig(image_path)
        print("Image saved to", image_path)
        plt.close()
    else:
        plt.show()


def plot_epsilon_pa_vs_sma_1stpeak(
    isolist,
    image_path=None,
    verbose=True
):
    t = isolist.to_table()

    smas = t['sma']
    epsilons = t['ellipticity']
    smas_kpc = smas * (2 * xlim / bins) # smas in kpc
    pas_deg = t['pa']  # position angle in degrees

    # Compute PA deviation from 0 or 180 degrees
    # Fix: handle Astropy Quantity with units
    if hasattr(pas_deg, 'value'):
        pa_deviation = np.minimum(np.abs(pas_deg.value), np.abs(180 - pas_deg.value))
    else:
        pa_deviation = np.minimum(np.abs(pas_deg), np.abs(180 - pas_deg))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Ellipticity
    ax1.plot(smas_kpc, epsilons, marker='o', linestyle='-', color='tab:blue', label='Ellipticity (ε)')
    ax1.set_xlabel('Semi-Major Axis (sma) [kpc]')
    ax1.set_ylabel('Ellipticity (ε)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    idx_peak = np.argmax(epsilons)
    ax1.axvline(smas_kpc[idx_peak], color='red', linestyle='--',
                label=f'Peak ε = {epsilons[idx_peak]:.3f} at {smas_kpc[idx_peak]:.2f} kpc')
    ax1.legend(loc='lower right')

    # Right y-axis: PA deviation
    ax2 = ax1.twinx()
    ax2.plot(smas_kpc, pa_deviation, 'x--', color='tab:orange', label='PA Deviation')
    ax2.set_ylabel('PA Deviation from 0° or 180° [deg]', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.legend(loc='upper right')

    plt.title('Ellipticity and PA Deviation vs Semi-Major Axis')
    fig.tight_layout()
    plt.grid(True)
    plt.tight_layout()

    if image_path:
        plt.savefig(image_path)
        if verbose:
            print("Image saved to", image_path)
        plt.close()
    else:
        plt.show()


def plot_epsilon_pa_vs_sma_2ndpeak(
    isolist,
    xlim,
    bins,
    image_path=None,
    verbose=True
):
    t = isolist.to_table()

    smas = t['sma']
    epsilons = t['ellipticity']
    smas_kpc = smas * (2 * xlim / bins)
    pas_deg = t['pa']

    # Compute PA deviation from 0 or 180 degrees
    if hasattr(pas_deg, 'value'):
        pa_deviation = np.minimum(np.abs(pas_deg.value), np.abs(180 - pas_deg.value))
    else:
        pa_deviation = np.minimum(np.abs(pas_deg), np.abs(180 - pas_deg))

    # --- Find all local maxima ---
    peaks, _ = find_peaks(epsilons)

    if len(peaks) < 2:
        print("[WARNING] Fewer than two peaks found in ellipticity.")
        second_peak_idx = peaks[0] if len(peaks) > 0 else None
    else:
        # Sort peaks by sma (i.e., radial order) and ignore the first (inner peak)
        peaks_sorted_by_sma = sorted(peaks, key=lambda i: smas[i])
        second_peak_idx = peaks_sorted_by_sma[1]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot ellipticity
    ax1.plot(smas_kpc, epsilons, marker='o', linestyle='-', color='tab:blue', label='Ellipticity (ε)')
    ax1.set_xlabel('Semi-Major Axis (sma) [kpc]')
    ax1.set_ylabel('Ellipticity (ε)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if second_peak_idx is not None:
        ax1.axvline(smas_kpc[second_peak_idx], color='red', linestyle='--',
                    label=f'2nd Peak ε = {epsilons[second_peak_idx]:.3f} at {smas_kpc[second_peak_idx]:.2f} kpc')
        ax1.legend(loc='lower right')

    # Plot PA deviation
    ax2 = ax1.twinx()
    ax2.plot(smas_kpc, pa_deviation, 'x--', color='tab:orange', label='PA Deviation')
    ax2.set_ylabel('PA Deviation from 0° or 180° [deg]', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.legend(loc='upper right')

    plt.title('Ellipticity and PA Deviation vs Semi-Major Axis')
    ax1.grid(True)
    fig.tight_layout()

    if image_path:
        plt.savefig(image_path)
        if verbose:
            print("Image saved to", image_path)
        plt.close()
    else:
        plt.show()


def select_bar_stars_no_bulge(snapshot, R_bar=5.0, q=0.4, R_inner=0.0, kinematic_cut=False, plot=False, view_range=7.0):
    """
    Select stars belonging to the bar region from a pynbody snapshot.
    
    Parameters
    ----------
    snapshot : pynbody.snapshot.SimSnap
        The simulation snapshot (already aligned so the bar is along the x-axis).
    R_bar : float, optional
        Maximum bar length in kpc (default: 5.0).
    q : float, optional
        Axial ratio (bar thickness relative to length). Smaller q → thinner bar (default: 0.4).
    R_inner : float, optional
        Inner cutoff radius to exclude bulge contamination in kpc (default: 0.0).
    kinematic_cut : bool, optional
        If True, filter stars by angular momentum (to remove transient disk stars).
    plot : bool, optional
        If True, generate a plot showing the bar star selection.
    view_range : float, optional
        Half-size of the plotting window in kpc (default: 7.0 → plot area ±7 kpc).

    Returns
    -------
    bar_stars : pynbody.filt.SubSnap
        A filtered SubSnap containing only bar stars.
    """
    stars = snapshot.stars

    # Extract coordinates
    x, y = stars['x'], stars['y']
    r = np.sqrt(x**2 + y**2)
    
    # Spatial mask for bar region
    mask_bar = (np.abs(y) < q*np.abs(x)) & (r < R_bar) & (r > R_inner)

    if kinematic_cut:
        # Angular momentum around z-axis
        jz = x * stars['vy'] - y * stars['vx']
        mask_bar &= (jz > 0)  # keep only prograde stars
    
    bar_stars = stars[mask_bar]

    # Optional plot
    if plot:
        plt.figure(figsize=(6, 6))
        #plt.scatter(stars['x'], stars['y'], s=0.1, alpha=0.2, label="All stars")
        plt.scatter(bar_stars['x'], bar_stars['y'], s=0.1, alpha=0.5, label="Bar stars")
        plt.gca().set_aspect('equal')
        plt.xlim(-view_range, view_range)
        plt.ylim(-view_range, view_range)
        plt.xlabel("x [kpc]")
        plt.ylabel("y [kpc]")
        plt.legend()
        plt.show()

    return bar_stars


def select_bar_stars_angmom(snapshot, R_bar=6.0, q=0.4, R_inner=0.0, kinematic_cut=False, disk_cut=True, view_range=7.0, plot=False, verbose=False):
    stars = snapshot.stars

    # Filter disk stars
    if disk_cut:
        # Use pynbody's circularity profile instead of crude mass/r estimate
        jz = stars['x'] * stars['vy'] - stars['y'] * stars['vx']
        r = np.sqrt(stars['x']**2 + stars['y']**2)

        # Calculate circular angular momentum at each radius
        v_circ = np.sqrt(np.abs(stars['phi']))  # Approximation from potential
        j_circ = r * v_circ

        epsilon = jz / (j_circ + 1e-8)
        disk_mask = epsilon > 0.7

        disk_stars = stars[disk_mask]
    else:
        disk_stars = stars

    if verbose:
        print("VERBOSE Total stars:", len(stars))
        print("VERBOSE Total disk stars:", len(disk_stars))

    # Bar selection
    x, y = disk_stars['x'], disk_stars['y']
    r = np.sqrt(x**2 + y**2)
    mask_bar = (np.abs(y) < q * np.abs(x)) & (r < R_bar) & (r > R_inner)

    if kinematic_cut:
        jz = x * disk_stars['vy'] - y * disk_stars['vx']
        mask_bar &= (jz > 0)

    bar_stars = disk_stars[mask_bar]

    if verbose:
        print("VERBOSE Total bar stars:", len(bar_stars))

    bar_fraction = len(bar_stars) / len(disk_stars) if len(disk_stars) > 0 else 0.0

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(disk_stars['x'], disk_stars['y'], s=0.1, alpha=0.2, label="Disk stars")
        plt.scatter(bar_stars['x'], bar_stars['y'], s=0.1, alpha=0.5, label="Bar stars")
        plt.gca().set_aspect('equal')
        plt.xlim(-view_range, view_range)
        plt.ylim(-view_range, view_range)
        plt.xlabel("x [kpc]")
        plt.ylabel("y [kpc]")
        plt.legend()
        plt.show()

    return bar_stars, disk_stars, bar_fraction


import pynbody
import numpy as np
from scipy.interpolate import interp1d

def select_bar_stars(snapshot, r_bar=6.0, bar_angle=0.0, disk_cut=True, epsilon_cut=0.7, verbose=True, plot=False):
    G = 4.30091e-6  # gravitational constant in (kpc km^2 / s^2 / Msun)

    # Align the bar
    stars = snapshot.stars
    if verbose:
        print("VERBOSE Total stars:", len(stars))

    # Compute cylindrical radius
    r = np.sqrt(stars['x']**2 + stars['y']**2)

    # Build profile for enclosed mass
    prof = pynbody.analysis.profile.Profile(snapshot, type='log', ndim=3)
    r_prof = prof['rbins']
    m_enc = np.cumsum(prof['mass'])  # enclosed mass in Msun

    # Compute circular velocity
    v_circ_prof = np.sqrt(G * m_enc / r_prof)

    # Interpolate circular velocity for each star
    v_circ_interp = interp1d(r_prof, v_circ_prof, bounds_error=False, fill_value=(v_circ_prof[0], v_circ_prof[-1]))
    v_circ = v_circ_interp(r)

    # Compute angular momentum and circularity
    jz = stars['x'] * stars['vy'] - stars['y'] * stars['vx']
    j_circ = r * v_circ
    epsilon = jz / (j_circ + 1e-8)

    if plot:
        plt.hist(epsilon, bins=100)
        plt.xlabel("epsilon")
        plt.ylabel("count")
        plt.show()

    # Filter disk stars
    if disk_cut:
        disk_mask = epsilon > epsilon_cut
        disk_stars = stars[disk_mask]
    else:
        disk_stars = stars

    if verbose:
        print("VERBOSE Total disk stars:", len(disk_stars))

    # Select bar stars within r_bar and bar-aligned region
    x_rot = disk_stars['x'] * np.cos(-bar_angle) - disk_stars['y'] * np.sin(-bar_angle)
    y_rot = disk_stars['x'] * np.sin(-bar_angle) + disk_stars['y'] * np.cos(-bar_angle)

    bar_mask = (np.abs(y_rot) < 0.5 * np.abs(x_rot)) & (np.sqrt(x_rot**2 + y_rot**2) < r_bar)
    bar_stars = disk_stars[bar_mask]

    if verbose:
        print("VERBOSE Total bar stars:", len(bar_stars))

    bar_fraction = len(bar_stars) / len(stars) if len(stars) > 0 else 0

    return bar_stars, disk_stars, bar_fraction

import numpy as np
import pynbody
from pynbody.analysis import profile
import matplotlib.pyplot as plt

def compute_disk_mask(snapshot, epsilon_cut=0.7, verbose=False):
    # Ensure physical units
    snapshot.physical_units()

    # Use total snapshot (stars+DM+gas) for circular velocity
    prof = profile.Profile(snapshot, type='log', ndim=3)
    r_prof = prof['rbins']                # [kpc]
    m_enc = prof['mass_enc']              # [Msun]

    # Gravitational constant in kpc (km/s)^2 / Msun
    G = 4.302e-6
    v_circ_prof = np.sqrt(G * m_enc / r_prof)  # km/s

    # Compute radius of each star
    r = np.sqrt(np.sum(snapshot.stars['pos'][:, :2]**2, axis=1))

    # Interpolate v_circ for each star
    v_circ = np.interp(r, r_prof, v_circ_prof)

    # Specific angular momentum (z-component)
    jz = snapshot.stars['j'][:, 2]

    # Circular angular momentum
    j_circ = r * v_circ

    # Epsilon parameter
    epsilon = jz / (j_circ + 1e-10)  # avoid division by zero

    # Plot epsilon vs radius for verification
    plt.figure(figsize=(8, 6))
    plt.scatter(r, epsilon, s=0.5, alpha=0.5)
    plt.axhline(y=epsilon_cut, color='r', linestyle='--', label=f'ε = {epsilon_cut}')
    plt.xlabel("Radius [kpc]")
    plt.ylabel("ε (jz/jcirc)")
    plt.title("Epsilon vs Radius")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Debug info
    if verbose:
        print(f"DEBUG:")
        print(f"  r range: {r.min():.3f} – {r.max():.3f} kpc")
        print(f"  v_circ range: {v_circ.min():.3f} – {v_circ.max():.3f} km/s")
        print(f"  jz range: {jz.min():.3e} – {jz.max():.3e}")
        print(f"  j_circ range: {j_circ.min():.3e} – {j_circ.max():.3e}")
        print(f"  epsilon range: {epsilon.min():.3f} – {epsilon.max():.3f}")

    # Disk mask
    disk_mask = epsilon > epsilon_cut
    if verbose:
        print(f"Disk stars selected: {disk_mask.sum()} / {len(snapshot.stars)}")

    return disk_mask, epsilon



if __name__ == '__main__':

    # Run pipeline
    print("######### SNAPSHOT ", fname, " #########")
    print("######### PARAMETER FILE ", paramfile, " #########")

    sn, density = load_snapshot_and_density(file, paramfile, bins, xlim, ylim)

    #isolist = fit_ellipses_to_density(density, bins, num_ellipses)

    #plot_density_with_ellipses_br(density, isolist, xlim, ylim, bins, image_path=image_dir + dens_image_fname)

    #plot_epsilon_pa_vs_sma_2ndpeak(isolist, xlim, bins, image_path=image_dir + ellips_image_fname)

    #bar_stars = select_bar_stars(sn, R_bar=6.0, q=0.4, R_inner=1.0, kinematic_cut=True, plot=True, view_range = 7)
    #print(f"Total {len(sn.stars)} simulation stars")
    #print(f"Selected {len(bar_stars)} bar stars")

    disk_mask, epsilon = compute_disk_mask(sn, epsilon_cut=0.7, verbose=True)
    disk_stars = sn.stars[disk_mask]

    print(f"Disk stars selected: {len(disk_stars)} / {len(sn.stars)}")

    bar_stars, disk_stars, bar_fraction = select_bar_stars(
        sn,
        r_bar=6.0
    )

    print(f"Disk stars: {len(disk_stars)}")
    print(f"Bar stars: {len(bar_stars)}")
    print(f"Bar fraction: {bar_fraction * 100:.2f}%")

