from photutils.isophote import Ellipse, EllipseGeometry
import Fourier_functions as mf
from Fourier_params import *
import matplotlib.pyplot as plt
import pynbody
import numpy as np

model = 'TNG50' # TNG50 skinny bar from Stuart Andersson
base_dir = '/home/ubuntu/projects/S2B/'
model_dir = base_dir + "models/" + model + "/"
image_dir = base_dir + "images/bar-shapes/" + model + "/"
fname = "455291_99"
image_fname = fname + ".png"
bins = 20

file = model_dir + fname
paramfile = model_dir + model + '.param'

def load_data(file = file, param = paramfile):
    print("----------- STEP 1 - LOADING DATA FROM THE FILE ---------------------")
    print("---")
    print("######### SNAPSHOT ", fname, " #########")
    print("######### PARAMETER FILE ", paramfile, " #########")

    sim = mf.pbload(file, paramfile, ncl_bar, bar_align, log)
    sim.physical_units()

    return sim

def faceon_density_to_nparray(snapshot, xlim=7, ylim=7, bins=20, verbose=True):

    """
    Generate  a face-on x-y number density map for stellar particles in a pynbody snapshot.

    Parameters:
        snapshot (pynbody snapshot): Loaded snapshot using pynbody.load().
        xlim (float): X-axis limit in kpc (centered).
        ylim (float): Y-axis limit in kpc (centered).
        bins (int): Number of bins along each axis.
        verbose (bool): If True, print diagnostic information.
    """

    # Center on galaxy and orient face-on using stellar angular momentum
    snapshot.physical_units()
    pynbody.analysis.angmom.faceon(snapshot.stars)

    if verbose:
        print("Snapshot centered and aligned face-on.")

    # Select x, y positions of stars
    x = snapshot.stars['x']
    y = snapshot.stars['y']

    # Compute 2D histogram (number density)
    density, xedges, yedges = np.histogram2d(
        x, y,
        bins=bins,
        range=[[-xlim, xlim], [-ylim, ylim]]
    )

    # Optional: transpose if needed depending on your visualization
    density = density.T  # to match imshow orientation (y rows, x cols)

    if verbose:
        print(f"Shape: {density.shape}, X range: {x.min():.1f}–{x.max():.1f}, Y range: {y.min():.1f}–{y.max():.1f}")

    return density

def ellipse_fitting(density):
    #Ellipse fitting - need to remove nans and infs
    density[np.where(np.isinf(density))] = 0
    density[np.where(np.isnan(density))] = 0

    #Initial guess
    xs = (bins + 1) / 2 # center position in pixels, i.e. bins
    zs = (bins + 1) / 2 # center position in pixels, i.e. bins
    sma = bins/2/2 # guess for the semimajor axis length in pixels
    eps = 0.2 # ellipticity
    # positon angle is defined in radians, counterclockwise from the
    # +X axis (rotating towards the +Y axis). Here we use 35 degrees 
    # as a first guess.
    pa = 5./180. * np.pi #position angle avoid 0.
    # pa = 35. / 180. * np.pi

    # note that the EllipseGeometry constructor has additional parameters with
    # default values. Please see the documentation for details.
    g = EllipseGeometry(xs, zs, sma, eps, pa)

    # the custom geometry is passed to the Ellipse constructor.
    ellipse1 = Ellipse(density, geometry=g) 

    #Ellipticity is 1 - (b/a), a = semi major axis
    #So the semi minor axis is a(1 - eps) 
    #Step controls how many steps along the semi major axis we go
    #Maximum sma should be the number of bins/2 which is usually 50
    #Add 1 to the maxsma to accommodate decimal step sizes like 30 ellipses

    #Temp fix num ellipses to get annuli along x 300 pc long
    # num_ellipses = int(np.round(x[keep].max()/.3, 0))
    num_ellipses = 30
    estep = bins/(num_ellipses-1)/2

    #We need to have num_ellipses-1 annuli so check this
    # isolist = ellipse1.fit_image(step=estep, maxsma=bins/2, 
    # linear=True) 
    #Leave a large maxgerr to maximise the chance of a fit being possible
    isolist = ellipse1.fit_image(step=estep, maxsma=bins/2, 
    linear=True, maxgerr = 2) 
    # smas = np.linspace(10, bins/2, 50)
    smas = isolist.to_table()['sma']
    print('Length of sma array {0}'.format(len(smas)))

    #If no meaningful fit...
    #if len(smas) == 0:
    #    continue

    return isolist, smas

def plot_ellipses_only(isolist, smas, bins, xlim, ylim, output_path):
    """
    Plot only the fitted ellipses (no background) and save the image.

    Parameters:
        isolist: Ellipse fitting result (IsophoteList)
        bins (int): Number of bins (image resolution)
        xlim, ylim (float): Axis limits in kpc
        output_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    xrange = xlim * 2
    yrange = ylim * 2

    smas = isolist.to_table()['sma']

    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y = iso.sampled_coordinates()
        x -= bins / 2
        y -= bins / 2
        x *= xrange / bins
        y *= yrange / bins
        ax.plot(x, y, color='grey', lw=1)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title("Fitted Ellipses (No Background)")

    plt.show()
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] Ellipse-only plot saved to {output_path}")

def plot_ellipses_with_density(density, isolist, smas, bins, xlim, ylim, output_path):
    """
    Plot face-on number density map with fitted ellipses overlay.

    Parameters:
        density (2D array): Face-on stellar density map (2D numpy array)
        isolist: Ellipse fitting result (IsophoteList)
        bins (int): Number of bins in density map
        xlim, ylim (float): Axis limits in kpc
        output_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the background number density
    extent = [-xlim, xlim, -ylim, ylim]
    ax.imshow(density, extent=extent, origin='lower', cmap='Greys', interpolation='nearest')

    xrange = xlim * 2
    yrange = ylim * 2
    smas = isolist.to_table()['sma']

    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y = iso.sampled_coordinates()
        x -= bins / 2
        y -= bins / 2
        x *= xrange / bins
        y *= yrange / bins
        ax.plot(x, y, color='red', lw=1)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title("Face-on Density with Fitted Ellipses")

    plt.show()
    plt.savefig(output_path)
    plt.close()
    print(f"[Saved] Ellipses with density plot saved to {output_path}")

if __name__ == '__main__':

    sn = load_data(file, paramfile)

    density = faceon_density_to_nparray(sn)

    isolist, smas = ellipse_fitting

    image_fname = image_dir + fname + 'ellips-fit.png'
    plot_ellipses_only(isolist, bins, xlim=7, ylim=7, output_path=image_fname)

    image_fname = image_dir + fname + 'dens-ellips-fit.png'
    plot_ellipses_with_density(density, isolist, bins, xlim=7, ylim=7, output_path=image_fname)