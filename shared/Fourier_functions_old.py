
# Plots number dencity for age groups
def plot_density_by_age(cmap,sim,pos,xlim,ylim,bins,snap,image_dir,e_list=False,save_file=True,show_plot=True,verbose_log=False):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit

    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
            grp_min_age = round(min(grp.star['age']),2)
            grp_max_age = round(max(grp.star['age']),2)
            print('***** Min stellar age in group - ' + str(grp_min_age) + ' Gyr.')
            print('***** Max stellar age in group - ' + str(grp_max_age) + ' Gyr.')
            if e_list:
                print("*** Bar ellipticity - ", e_list[age_grp-1])
        
        #Extract phase space data for the model for stars in the group
        z_, x_, y_ = grp.star['z'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        dfg_stat2d,dfg_xedges,dfg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, z_,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(dfg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                image = axes[i].imshow(stat2d_lst[i], 
                            origin = 'lower',
                            extent = [-xlim, xlim, -ylim, ylim ],
                            norm = LogNorm(),
                            cmap = cmap)
                xcent = (dfg_xedges[1:] + dfg_xedges[:-1]) / 2
                ycent = (dfg_yedges[1:] + dfg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, np.log10(stat2d_lst[i]), linewidths = 0.5, linestyles = 'dashed', colors = 'k')
                if e_list:
                    axes[i].title.set_text("Age group " + str(i+1) + " e = " + str(e_list[i]))
                else:
                    axes[i].title.set_text("Age group " + str(i+1))
                circle1 = plt.Circle((0, 0), 0.5, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 0.75, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " num density.")
    plt.setp(axes[:], xlabel = 'x [kpc]')
    plt.setp(axes[0], ylabel = 'y [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + pos + '_density_by_age_3grp_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots bar ellipticity
def bar_ellipticity_by_age(sim,xlim,ylim,bins,snap,image_dir,save_file=True,show_plot=True,verbose_log=True):

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    # List of ellipticities and radii per age group for plotting
    e_list = [[],[],[]]
    r_list = [[],[],[]]
    e_age_grp = []

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))

        #Extract phase space data for the model for stars in the group
        z_, x_, y_ = grp.star['z'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population in the group
        dfg_stat2d,dfg_xedges,dfg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, z_,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)

        unit = 2*xlim/bins
        if verbose_log:
            print("Unit in kpc:", round(unit,2))

        e = ellipticity_quadrupole(unit,bins,dfg_stat2d)
        print('*** Age group', age_grp, 'ellipticity:', e)
        e_age_grp.append(e)

        if verbose_log:
            for radius in range(1,int(bins/2)+1):
                sub_list = extract_sublist(dfg_stat2d, radius)
                e = ellipticity_quadrupole(unit,2*radius,sub_list)
                e_list[age_grp-1].append(e)
                r_list[age_grp-1].append(round(radius*unit,2))
                if verbose_log:
                    print('*** Radius:', round(radius*unit,2), " kpc, ", 'ellipticity:', e)

        x_panels = 3
        y_panels = 1
        figsize_x = 3*x_panels      # inches
        figsize_y = 3.5*y_panels    # inches

        # Make the figure and sub plots
        fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
        for i in range(x_panels):
            image = axes[i].plot(r_list[i],e_list[i])
            axes[i].title.set_text("Age group " + str(i+1))
            # axes[i].set_ylim(min(e_list[age_grp-1]), max(e_list[age_grp-1]))

            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            #cbar = fig.colorbar(image, cax=cax, orientation='vertical')
            #cbar.set_label(cbar_label_lst[i])
            if i > 0:
                axes[i].set_yticklabels([])

            fig.tight_layout()
            fig.suptitle(snap.replace(".gz","") + " bar ellipticity per age group.")
            plt.setp(axes[:], xlabel = 'R [kpc]')
            plt.setp(axes[0], ylabel = 'e')

            if save_file:
                image_name = image_dir + snap.replace(".gz","") + '_ellipticity_by_age_3grp_' + str(xlim) + 'kpc' + '.png'
                plt.savefig(image_name)
                print("Image saved to",image_name)
            else:
                print("Image saving is turned off.")
            if show_plot:
                plt.show()
            else:
                print("On-screen ploting is turned off.")

    return e_age_grp


# Wrap function for age groups, calls bar_length_Fm
def bar_length_by_age_Fm(sim,Fm,snap,image_dir,save_file=True):

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
    
        bar_length_Fm(grp,Fm,snap,image_dir,age_grp,save_file)

    return None


# Plots edge-on sigma for age groups
def plot_sigma_by_age(cmap,sim,xlim,ylim,bins,snap,image_dir,save_file=True,show_plot=True,verbose_log=False):

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    stat2d_lst = []

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
    
        #Extract phase space data from the model for the stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        vdg_stat2d,vdg_xedges,vdg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(vdg_stat2d.T)

    for i in range(x_panels):
                image = axes[i].imshow(stat2d_lst[i], 
                            origin = 'lower',
                            extent = [-xlim, xlim, -ylim, ylim ],
                            cmap = cmap)
                xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
                ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, stat2d_lst[i], linewidths = 2, linestyles = 'dashed', colors = 'w')
                axes[i].title.set_text("Age group " + str(i+1))
                # For nuclear bars teh circle R is 0.5 and 0.75 
                circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " LOS velocity dispersion.")
    plt.setp(axes[:], xlabel = 'y [kpc]')
    plt.setp(axes[0], ylabel = 'z [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_sigma_by_age_3grp_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots edge-on sigma for age groups using the unsharp mask
def plot_sigma_by_age_umask(cmap,sim,xlim,ylim,bins,blur,snap,image_dir,save_file=True,show_plot=True,verbose_log=False):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
    
        #Extract phase space data for the model for stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        vdg_stat2d,vdg_xedges,vdg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(vdg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                
                # Apply Unsharp Mask
                blurred = gf(stat2d_lst[i], sigma=blur)  # Apply Gaussian blur
                sharpened = stat2d_lst[i] + (stat2d_lst[i] - blurred)  # Unsharp masking

                # Plot the sharpened image
                image = axes[i].imshow(sharpened,
                      origin='lower',
                      extent=[-xlim, xlim, -ylim, ylim],
                      cmap=cmap)

                xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
                ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, stat2d_lst[i], linewidths = 2, linestyles = 'dashed', colors = 'w')
                axes[i].title.set_text("Age group " + str(i+1))
                # For nuclear bars the circle R is 0.5 and 0.75 
                circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " LOS velocity dispersion (sharp " + str(blur) + ").")
    plt.setp(axes[:], xlabel = 'y [kpc]')
    plt.setp(axes[0], ylabel = 'z [kpc]')

    if save_file:
        image_name = f"{image_dir}{snap.replace('.gz', '')}_sigma_by_age_3grp_{xlim}kpc_umask_b{blur}.png"
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots edge-on sigma for age groups using grouping switch and unsharp mask
def plot_sigma_by_age_grp_sw_umask(cmap,sim,blur,snap,image_dir,save_file=True):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")
    
    # Check if min age is not equal to max age    
    check_ages(sim)
            
    # Number density statistics per age group
    age_grp = 0

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        if verbose_log:
            print_ages(grp,"group")
    
        #Extract phase space data for the model for stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        vdg_stat2d,vdg_xedges,vdg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(vdg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                
                # Apply Unsharp Mask
                blurred = gf(stat2d_lst[i], sigma=blur)  # Apply Gaussian blur
                sharpened = stat2d_lst[i] + (stat2d_lst[i] - blurred)  # Unsharp masking

                # Plot the sharpened image
                image = axes[i].imshow(sharpened,
                      origin='lower',
                      extent=[-xlim, xlim, -ylim, ylim],
                      cmap=cmap)

                xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
                ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, stat2d_lst[i], linewidths = 2, linestyles = 'dashed', colors = 'w')
                axes[i].title.set_text("Age group " + str(i+1))
                # For nuclear bars teh circle R is 0.5 and 0.75 
                circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    suptit_name = snap.replace(".gz","") + " LOS velocity dispersion " + grp_sw + " (sharp " + str(blur) + ")."
    fig.suptitle(suptit_name)
    plt.setp(axes[:], xlabel = 'y [kpc]')
    plt.setp(axes[0], ylabel = 'z [kpc]')

    if save_file:
        image_name = f"{image_dir}{snap.replace('.gz', '')}_sigma_by_age_3grp_{xlim}kpc_umask_b{blur}_{grp_sw}.png"
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")

    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# This function calculates a bar ellipticity using ellipse fit method
def ellipticity_from_ellipse_fit(density, unit):
    center = density.shape[0] // 2
    geometry = EllipseGeometry(x0=center, y0=center, sma=5, eps=0.1, pa=0)
    ellipse = Ellipse(density, geometry)  # ← pass raw array directly
    isolist = ellipse.fit_image()

    if isolist is None or len(isolist) == 0:
        if verbose_log:
            print("Isolist is None!")
        return [], []

    radii = [iso.sma * unit for iso in isolist]
    ellipticities = [iso.eps for iso in isolist]

    return radii, ellipticities

