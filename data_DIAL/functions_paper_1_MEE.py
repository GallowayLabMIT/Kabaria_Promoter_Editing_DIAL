# Import packages
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pathlib import Path
import rushd as rd
import scipy as sp
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from statannot import add_stat_annotation


#Setting the style of all seaborn plots to paper here. If there is a fontsize, linewidth, etc. in the function itself, this can override the parameters defined by paper
sns.set_context(context='paper')



def custom_density_plot(xcat=0,ycat=0,hue=None,data=None,points = 1e6, hue_order=None,
                        savetitle='',plottitle='',palette='muted',xlim=(0.5*10**0, 1*10**6),ylim = (0.5*10**0, 1*10**5),
                        type=['scatter'],alpha=0.2, legend='auto', figsize=(3,2),
                        hline=False,              # Horizontal line to be added to the plot (float)
                        vline=False               # Vertical line to be added to the plot (float)
                        ):
    
    # Ensure only necessary data is in the dataframe
    #data = data[data.Condition.isin(hue_order)]

    # reorder dataframe based on hue_order for plotting order purposes
    #data.Condition = pd.Categorical(data.Condition,ordered=True,categories=hue_order) 
    #data_sort = data.sort_values('Condition')

    #Down sample by hue 
    data_per_cond_min = data.groupby([hue])['FSC-A'].count().min()
    min_num = min(data_per_cond_min,points)
    print(min_num)
    #display(data_per_cond_min)
    #display(data.groupby([hue])['FSC-A'].count().min())
    data = data.groupby([hue]).sample(n=min_num, random_state=1) # pulls number samples that is 'points' or less depending on sample 

    g = sns.JointGrid(data=data, x=xcat, y=ycat, hue=hue, xlim=xlim, ylim=ylim, ratio=3, space=0)
    g = g.plot_marginals(sns.kdeplot, fill=True, alpha=alpha, common_norm=False, log_scale=True, palette=palette,hue_order=hue_order)
    ax = g.ax_joint
    ax.set_xscale('log') #log
    ax.set_yscale('log')
    g.ax_marg_x.set_xscale('log')
    g.ax_marg_y.set_yscale('log')

    if 'scatter' in type:
        g = g.plot_joint(sns.scatterplot,s=3,palette=palette, hue_order=hue_order) #scatter plot in middle graph
    
    if 'contour' in type:
        g = g.plot_joint(sns.kdeplot, alpha=0.8,palette=palette, hue_order=hue_order) #contour plot in middle graph 
    
    if 'filled contour' in type:
        g = g.plot_joint(sns.kdeplot, alpha=0.6,shade=True,palette=palette, hue_order=hue_order) #filled contour plot in middle graph 

    if legend == True:
        sns.move_legend(g.ax_joint, title='Condition', loc='upper left', bbox_to_anchor=(1.35,1), frameon=False)
    if legend == False: 
        g.ax_joint.legend_.remove()

    #Add lines
   
    if vline != False:
        for i in vline:
            ax.axvline(i,0,1,ls='--',lw=0.7,color='black',alpha=0.7)
    if hline != False:
        ax.axhline(hline,0,1,ls='--',lw=0.7,color='black',alpha=0.7)

    #add overall title
    g.fig.suptitle(plottitle)

    # ax.set_xlabel(xcat, fontsize=22)  # Adjust the font size of x-axis label
    # ax.set_ylabel(ycat, fontsize=22)  # Adjust the font size of y-axis label

    # ax.tick_params(axis='both', labelsize=22,length=5)  # Adjust the font size of tick labels and tick lengths

    plt.show(g)
    return g 

def custom_hist_plot( data_now, parameters, hue=None, palette='flare', hueorder=None, xsize=15, ysize=3, title='', 
                     alpha=0.1,legendtitle='Condition', line=[], legend=True,
                     xlim = False, fontsize = None, axis_lines={'top': True, 'bottom': True, 'left': True, 'right': True}, 
                     data_neg = [], #Add in a negative curve by making this dataset non False
                     linewidth=True): #Added a linewidth parameter
    
    parameterslen = len(parameters) #count number of parameters you want plotted as histograms
    
    
    if parameterslen > 1:
        fig, axes = plt.subplots(1, parameterslen, figsize=(xsize,ysize))
        #Add title
        fig.suptitle(title)
        #iterate through parameters
        for i in range(parameterslen):
            if i < (parameterslen-1):
                fig = sns.kdeplot(ax=axes[i],data=data_now,x=parameters[i],hue=hue,log_scale=True,legend=False,fill=True, alpha = alpha,common_norm=False,palette=palette,hue_order=hueorder,linewidth=linewidth)
                if len(data_neg) >0:
                    print('unstained should show')
                    fig = sns.kdeplot(ax=axes[i],data=data_neg,x=parameters[i], hue=None, log_scale=True,legend=False,shade=False, alpha = 1,
                                      common_norm=False, color='black',linestyle='--',linewidth=linewidth)
                xl = parameters[i] #xlabel
                yl = 'Density' #ylabel 
                if xlim != False:
                    fig.set_xlim(xmin=xlim[0], xmax=xlim[1])
                if fontsize != None:
                    fig.set_xlabel(xl, fontsize=fontsize)
                    fig.set_ylabel(yl, fontsize=fontsize)
                    fig.tick_params(axis='both', labelsize=fontsize,length=5)
                #Despine
                sns.despine(ax=axes[i], top=True, right=True, left=False, bottom=False)
            elif i == (parameterslen-1):
                fig = sns.kdeplot(ax=axes[i],data=data_now,x=parameters[i],hue=hue,log_scale=True,fill=True, alpha = alpha,common_norm=False,palette=palette,hue_order=hueorder, legend=legend,linewidth=linewidth) 
                xl = parameters[i]
                yl = 'Density'
                if len(data_neg) >0:
                    print('unstained should show')
                    fig = sns.kdeplot(ax=axes[i],data=data_neg,x=parameters[i], hue=None, log_scale=True,legend=False,shade=False, alpha = 1,
                             common_norm=False, color='black',linestyle='--',linewidth=linewidth)
                if fontsize != None:
                    fig.set_xlabel(xl, fontsize=fontsize)
                    fig.set_ylabel(yl, fontsize=fontsize)
                    fig.tick_params(axis='both', labelsize=fontsize,length=5)
                #Despine
                sns.despine(ax=axes[i], top=True, right=True, left=False, bottom=False)
        if legend==True and hue != None:
            sns.move_legend(axes[-1], title=legendtitle, loc='upper left', bbox_to_anchor=(1,1), frameon=False)
    elif parameterslen == 1:
            fig = sns.kdeplot(data=data_now,x=parameters[0],y=None, hue=hue,log_scale=True,fill=True, alpha = alpha,common_norm=False,palette=palette,hue_order=hueorder, legend=legend,linewidth=linewidth) #reporter and EGFP 
            xl = parameters[0]
            yl = 'Density'
            if len(data_neg) >0:
                    print('unstained should show')
                    fig = sns.kdeplot(data=data_neg,x=parameters[0], hue=None, log_scale=True,legend=False,shade=False, alpha = 1,
                                      common_norm=False, color='black',linestyle='--',linewidth=linewidth)
            if fontsize != None:
                fig.set_xlabel(xl, fontsize=fontsize)
                fig.set_ylabel(yl, fontsize=fontsize)
                fig.tick_params(axis='both', labelsize=fontsize,length=5)
            # Add title
            fig.set_title(title)
            if legend == True and hue != None:
                sns.move_legend(fig, title=legendtitle, loc='upper left', bbox_to_anchor=(1,1), frameon=False)
            
            ##If single hist doesnt work anymore its prob bc of the two lines below
            ax = plt.gca()
            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False) #Despine
            ### If single hist doesnt work anymore its prob bc of the two lines above
    
    if bool(line):
        plt.axvline(line,0,1,ls='--',lw=0.7,color='black',alpha=0.7)

    #Set xlim
    if xlim != False:
        fig.set_xlim(xmin=xlim[0], xmax=xlim[1])

    # Remove axis lines
    if not axis_lines['top']:
        fig.spines['top'].set_visible(False)
    if not axis_lines['right']:
        fig.spines['right'].set_visible(False)
    if not axis_lines['bottom']:
        fig.spines['bottom'].set_visible(False)
    if not axis_lines['left']:
        fig.spines['left'].set_visible(False)


    plt.show(fig)
    return fig

def rename(index):
    if index[1] == '': return index[0]
    else: return index[0] + '_' + index[1]

def calc_stats(df,by,x,stat):
    
    # Filter data to remove log-unfriendly values
    d = df.copy()
    for xi in x: d = d.loc[(d[xi]>0)]

    # Group and compute stats
    grouped = d.groupby(by=by)
    stats = grouped[x].agg(stat).reset_index()

    # Rename columns as 'col_stat'
    stats.columns = stats.columns.map(lambda i: rename(i))

    # Add columns for count and fraction (of total one grouping level up)
    s = grouped[x[0]].count()
    if len(by) > 1:
        s = (s/s.groupby(by[:-1]).transform('sum')).dropna().reset_index(name='Fraction')
        stats = stats.assign(Fraction=s['Fraction'])
    s['Count'] = grouped[x[0]].count().rename('Count').reset_index()['Count']
    stats = stats.assign(Count=s['Count'])

    return stats

def summary_plot( x, y, hue, data,pairs=[], palette = 'muted', order=None, hue_order =None, plottitle="",savetitle = '',
                 save_fig=False, dodge=True,x_rot=0, scientific=True,yscale="linear", stat_text = 'star', 
                 ylim=False,aspect=None, plot_reps=True, join=False,
                 xlabel = 'Default', ylabel = 'Default', legend=True,
                 type = 'pointplot', figsize = None, markers=True,
                 fontsize=False, axis_lines={'top': False, 'bottom': True, 'left': True, 'right': False},capsize=True):


        if aspect != None:
            g, ax = plt.subplots()
            ax.set_box_aspect(aspect)
            #plt.rc('ytick', labelsize=15) 
        if figsize != None:
            plt.figure(figsize=figsize)

        #plot all of the replicates of each condtion
        if plot_reps == True:
            g = sns.stripplot(data=data, x=x, y=y,hue=hue, dodge = dodge, alpha=0.2, palette=palette,size=4,hue_order=hue_order, order=order)  ;

        #Add in the means 
        #sns.barplot(x="reporter", y="eGFP-A_gmean",hue='rec', data=s, estimator=np.mean,capsize=.1,alpha=0.2,errwidth=2,errcolor='gray')
        if dodge == True and hue is not None:
            n_hue_levels = len(data[hue].unique())
            if n_hue_levels > 1:
                dodge_amt = 0.5
            else:
                dodge_amt = False
        elif dodge == True and hue is None:
            dodge_amt = 0.5
        else:
            dodge_amt = False

        

        if type == 'pointplot':
            g = sns.pointplot(x=x, y=y,hue=hue, data=data, estimator=np.mean, 
                        errorbar='se', capsize=0.1,errwidth=0.75, join=join,scale=0.75, dodge = dodge_amt , 
                        palette=palette,hue_order=hue_order, order=order, markers=markers)
            if legend==False:
                g.get_legend().remove()
                #Old pointpolt settings 
            # g = sns.pointplot(x=x, y=y,hue=hue, data=data, estimator=np.mean, 
            #     errorbar='se', capsize=.1,errwidth=1, join=join, markers='_',scale=1.5, dodge = dodge_amt , 
            #     palette=palette,hue_order=hue_order, order=order);
        if type == 'barplot':
            g = sns.barplot(x=x, y=y, hue=hue, data=data, estimator=np.mean,
                            errorbar='se', capsize=.1,errwidth=1, dodge = dodge_amt , 
                    palette=palette, hue_order=hue_order, order=order, alpha = 0.7,markers=markers)
        if legend==True and hue != None:
            sns.move_legend(g, title='Condition', loc='upper left', bbox_to_anchor=(1,1), frameon=False);
    
        
        #rotate the x-labels
        labels = g.get_xticklabels()
        g.set_xticklabels(labels=labels, rotation=x_rot, ha='right')

        #set the y-scale
        plt.yscale(yscale)

        #Set ylim
        if ylim != False:
            g.set_ylim(ymin=ylim[0], ymax=ylim[1])

        #make scientific axes 
        if scientific != False:
            if yscale == "linear":
                plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

        #Add in the T-tests
        if len(pairs)>0:
            add_stat_annotation(g, data=data, x=x, y=y,hue=hue,
                        box_pairs=pairs, order=order, hue_order=hue_order,
                        test='t-test_ind', text_format=stat_text, loc='inside', verbose=2, 
                        comparisons_correction= None)
        
        #set x-label and y-label 
        xl = xlabel; yl = ylabel; 
        if xlabel == 'Default': 
             xl = x
        if ylabel == 'Default': 
             yl = y
        plt.xlabel(xl)  # Adjust x-axis label
        plt.ylabel(yl)  # Adjust y-axis label

        if fontsize != False: 
            g.set_xlabel(xl, fontsize=fontsize)  # Adjust the font size of x-axis label
            g.set_ylabel(yl, fontsize=fontsize)  # Adjust the font size of y-axis label
            #set ticks 
            g.tick_params(axis='both', labelsize=fontsize,length=5)  # Adjust the font size of tick labels and tick lengths


        plt.title(plottitle)
        g.spines['top'].set_visible(True)
        g.spines['right'].set_visible(True)
        g.spines['bottom'].set_visible(True)
        g.spines['left'].set_visible(True)

        if not axis_lines['top']:
            g.spines['top'].set_visible(False)
        if not axis_lines['right']:
            g.spines['right'].set_visible(False)
        if not axis_lines['bottom']:
            g.spines['bottom'].set_visible(False)
        if not axis_lines['left']:
            g.spines['left'].set_visible(False)
        
        plt.show(g)
        
        # adjust figure size to accommodate all axes decorations
        plt.tight_layout()

        #save figure
       
        h = g.get_figure()
        if save_fig==True:
                h.savefig('./figs/' + savetitle + '.svg',dpi=300,bbox_inches='tight')
        return g

def scatter_plot(data=None, *, x=None, y=None, hue=None, 
                 size=None, style=None, palette='bright', 
                 hue_order=None, hue_norm=None, 
                 sizes=None, size_order=None, size_norm=None, 
                 markers=True, style_order=None, legend='auto', ax=None, 
                 yscale = 'log', xscale = 'log', ylim=False, xlim=False, 
                 plottitle = '',
                 s=11, 
                 xlabel = 'Default', ylabel = 'Default', 
                 #fontsize=14,
                 #labelsize = 14, 
                 std_x =None, std_y = None, #Enter the column names that contain teh standard deviations 
                 new_figure = False, 
                 **kwargs
                 ): 
    
    g = sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, style=style, palette=palette,
                      hue_order=hue_order, hue_norm=hue_norm, sizes=sizes, 
                      size_order=size_order, size_norm=size_norm, markers=markers, 
                      style_order=style_order, legend=legend, ax=ax, 
                      s=s)
    
    if legend=='auto' and hue != None:
           sns.move_legend(g, title='Condition', loc='upper left', bbox_to_anchor=(1,1), frameon=False)

    #Add in standard deviation bars if applicable
    ax = plt.gca()
    if std_x != None:
        if std_y !=None: 
            ax.errorbar(x=data[x], y=data[y],
                xerr=data[std_x], yerr=data[std_y],
                color='None', ecolor='black', elinewidth=1, zorder=1)
        else: 
            ax.errorbar(x=data[x], y=data[y],
                xerr=data[std_x], yerr=None,
                color='None', ecolor='black', elinewidth=1, zorder=1)
    elif std_y !=None:
        ax.errorbar(x=data[x], y=data[y],
                xerr=None, yerr=data[std_y],
                color='None', ecolor='black', elinewidth=1, zorder=1)


    #Despine
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    #set the y-scale, x-scale
    plt.yscale(yscale)
    plt.xscale(xscale)

    #set x-label and y-label 
    xl = xlabel; yl = ylabel; 
    if xlabel == 'Default': 
         xl = x
    if ylabel == 'Default': 
         yl = y
    #(I commented this out when setting the sns = paper because it was overriding it)
    #g.set_xlabel(xl, fontsize=fontsize)  # Adjust the font size of x-axis label
    #g.set_ylabel(yl, fontsize=fontsize)  # Adjust the font size of y-axis label


    #Set ylim
    if ylim != False:
        g.set_ylim(ymin=ylim[0], ymax=ylim[1])
    #Set xlim
    if xlim != False:
        g.set_xlim(xmin=xlim[0], xmax=xlim[1])

    #set ticks (I commented this out when setting the sns = paper because it was overriding it)
    #g.tick_params(axis='both', labelsize=labelsize,length=5)  # Adjust the font size of tick labels and tick lengths


    plt.title(plottitle)

    return g 



##def summary_plot_scatter(x, y, hue, data,pairs=[], palette = 'muted', order=None, hue_order =None, plottitle="",savetitle = '',save_fig=False, dodge=True,x_rot=0, scientific=True,yscale="linear", stat_text = 'star'):
             #plot all of the replicates of each condtion

def hex_to_rgb(hex_code):
    """Convert hex color to RGB."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def create_custom_colormap(hex_colors=['#050423', '#0D2CA0', '#042ED4', '#1BA7E3', '#20D5DE', '#6CF0F5', '#A2F1E4'],
                            num_colors=9):
    """Create a custom colormap with specified number of colors from hex color list."""
    # Convert hex colors to RGB
    colors = [hex_to_rgb(color) for color in hex_colors]
    
    positions = np.linspace(0, 1, len(colors))  # Equally spaced positions for original colors
    
    # Interpolate colors
    new_positions = np.linspace(0, 1, num_colors)
    new_colors = np.array([np.interp(new_positions, positions, [c[0] for c in colors]),
                           np.interp(new_positions, positions, [c[1] for c in colors]),
                           np.interp(new_positions, positions, [c[2] for c in colors])]).T
    
    # Convert to list of RGB tuples
    rgb_tuples = [(r, g, b) for r, g, b in zip(new_colors[:, 0], new_colors[:, 1], new_colors[:, 2])]
    
    return rgb_tuples