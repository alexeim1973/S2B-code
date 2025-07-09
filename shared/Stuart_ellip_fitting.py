from photutils.isophote import Ellipse, EllipseGeometry

#Ellipse fitting - need to remove nans and infs
density[np.where(np.isinf(density))] = 0
density[np.where(np.isnan(density))] = 0

 #Initial guess
xs = (bins_xz + 1) / 2 # center position in pixels, i.e. bins
zs = (bins_xz + 1) / 2 # center position in pixels, i.e. bins
sma = bins_xz/2/2 # guess for the semimajor axis length in pixels
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

estep = bins_xz/(num_ellipses-1)/2

 #We need to have num_ellipses-1 annuli so check this
 # isolist = ellipse1.fit_image(step=estep, maxsma=bins_xz/2, 
 # linear=True) 
 #Leave a large maxgerr to maximise the chance of a fit being possible
isolist = ellipse1.fit_image(step=estep, maxsma=bins_xz/2, 
linear=True, maxgerr = 2) 
 # smas = np.linspace(10, bins/2, 50)
smas = isolist.to_table()['sma']
print('Length of sma array {0}'.format(len(smas)))

 #If no meaningful fit...
if len(smas) == 0:
    continue

 #Plot the ellipses
for sma in smas:
    iso = isolist.get_closest(sma)
    x1, y1, = iso.sampled_coordinates()
    x2 = x1 - bins_xz/2
    y2 = y1 - bins_xz/2
    x2 *= xrange/bins_xz
    y2 *= yrange/bins_xz
    ax1.plot(x2, y2, color='grey', lw=1)

    eps = isolist.to_table()['ellipticity']
    x0 = isolist.to_table()['x0']
    y0 = isolist.to_table()['y0']

    #Convert to pairs of elliptical annuli
    annuli = np.array(list(zip(smas[::1], smas[1:])))
    ellipticities = np.array(list(zip(eps[::1], eps[1:])))
    x0s = np.array(list(zip(x0[::1], x0[1:])))
    y0s = np.array(list(zip(y0[::1], y0[1:])))