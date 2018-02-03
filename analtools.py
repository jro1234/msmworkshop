'''
TODO 
Check out Sebastian's PySDA package to see
if it can be used for feature selection.

This file contains functionality to process
bodies of data for PyEMMA and other analysis,
and then create plots to examine the effect
of choosing different lag times in TICA and
MSM creation.

By using a sufficient number of cluster
centers in defining microstates, the analysis
should be insensitive to this paramter. 

'''

import numpy as np
import multiprocessing
import itertools
import pyemma

import types


def dump_data(filepath, data, format_template, top_line=None, chunksize=1000):

    datablock = ''
    if top_line:
        datablock += top_line + '\n'

    with open(filepath, 'w') as out:
        for i, line in enumerate(data):

            datablock += format_template.format(*line) + '\n'

            if i % chunksize == 0 and i > 0:
                out.write(datablock)
                datablock = ''
        else:
            out.write(datablock)



def multiproc(funcname, data, *args, **kwargs):
    #TODO N won't work
    '''
    data :: [list-like]

    if an object with a length and no kwargs 'N'
    is given, each item will be
    mapped to `func` in a separate process.
    otherwise, the same data will be given to
    N processes

    funcname :: <str> or <function>
    name of a function in analtools namespace
    if string, otherwise a function.
    '''
    if isinstance(funcname, str):
        func = globals()[funcname]
    elif isinstance(funcname, (types.FunctionType, types.BuiltinFunctionType)):
        func = funcname

    if 'N' in kwargs:
        N = kwargs['N']
    else:
        N = len(data)

    pool = multiprocessing.Pool(N)

    print("function: ", func)
    print("args: ", args)
    print("kwargs: ", kwargs)
    if args or kwargs:
        lambit = lambda dat: pool.apply_async(
            func, args=[dat]+list(args), kwds=kwargs)

    else:
        lambit = lambda dat: pool.apply_async(
            func, dat)

    result = [p.get() for p in map(lambit, data)]

    return result



def plot_energy_sampleddist(ticas, plt):
    fig, axes = plt.subplots(3,3, figsize=(10,10),)# sharex=True,)# sharey=True)
    axes = np.hstack(axes)
    for i, ystacked_tica in enumerate(ticas['ystacked_ticas']):
        ax = axes[i]
        if isinstance(ystacked_tica, list):
            ystacked_tica = np.concatenate(ystacked_tica)

        tica0_counts, tica0_coords = np.histogram(ystacked_tica[:,0],bins=32)

        #plt.plot(tica0_coords[1:], tica0_counts)
        energy_shift = np.log(max(tica0_counts))
        ax.plot(tica0_coords[1:], np.divide(tica0_counts, 1000, ), color='blue')#ystacked_tica.shape[0]))
        ax.plot(tica0_coords[1:], energy_shift + -np.log(tica0_counts), color='green')
        #print(tica0_counts[0]/ystacked_tica.shape[0])
        ax.set_title(ticas['ticas'][i].describe().split(',')[1].split(';')[0].strip())
        #plt.plot(egv,color=plt.cm.winter(20*i))  

    fig.suptitle("TICA Slowest Process at Different Lags",size=20,y=1.05)
    fig.tight_layout()



def plot_free_energy(ystacked_tica, plt, lagstring):
    pairs = tuple(itertools.combinations(range(ystacked_tica.shape[1]), 2))

    fig, axes = plt.subplots(1,3, figsize=(10,4), sharex=True, sharey=True)
    fig.suptitle('TIC Energy Landscape for Lag of %s ' % lagstring)
    for ax, (i, j) in zip(axes, pairs[:]):
        pyemma.plots.plot_free_energy(ystacked_tica[:, i], ystacked_tica[:, j], nbins=64, ax=ax, cbar=False)
        ax.set_title("%i-%i" % (i,j))
    fig.tight_layout()



#Compare the ITS over TICA calculation lag orders of 0 (shown), 1, and 2
def plot_msm_its(microstates, plt, lag):
        fig, axes = plt.subplots(1,1, figsize=(10,5), sharex=True, sharey=True)
        its_lags = [1, 5, 10, 25, 50, 100]#, 250]
        #for i,ax in enumerate(np.hstack(axes)):
        its = pyemma.msm.its(microstates.dtrajs, lags=its_lags)
        pyemma.plots.plot_implied_timescales(its, ax=axes)

        fig.suptitle("ITS using TICA lag {0}".format(lag))



def plot_msm_timescales(msms, plt):
    # subplots to get the legend handle
    # and reverse order
    fig, ax = plt.subplots(1)
    n_vals = len(msms)
    for i,msm in enumerate(msms):
        #ax.plot(msms[i].timescales(),
        ax.plot(msm.timescales(),
                linewidth=1,
                label='TICA lag %d' % msm.lag,
                color=plt.cm.plasma(25*(n_vals-i)),
                #marker='o'
               )
    plt.xlabel('index')
    plt.ylabel(r'timescale (1 $\mu$s)')
    plt.xlim(-0.5,10.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right')
    #plt.legend()



def plot_eigenvectors(ticas, plt):
    lambtitle = lambda tica: tica.describe().split(',')[1].split(';')[0].strip()
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True,)# sharey=True)
    for j,ax in enumerate(np.hstack(axes)):
        tica = ticas[j]
        if j%3==0:
            ax.set_ylabel("Feature Weights")
        if j>5:
            ax.set_xlabel("Feature Index")
        #print(tica.describe())
        n_vals = tica.eigenvectors.shape[0]
        for i,egv in enumerate(reversed(tica.eigenvectors)):
            ax.plot(egv,color=plt.cm.winter(20*(n_vals-i)))
            ax.set_title(lambtitle(tica))

    fig.suptitle("TICA Eigenvectors at Different Lags",size=20)



def plot_eigenvalues(ticas, plt, ax=None):
    lagsteps = lambda tica: tica.describe().split(',')[1] \
                             .split(';')[0].split('=')[1].strip()

    xvals = range(1,ticas[0].eigenvalues.shape[0]+1)
    for i,tica in enumerate(ticas):
        ax.plot(xvals,
                 abs(tica.eigenvalues),
                 color = plt.cm.winter(25*i),
                 label = 'lag %s' % lagsteps(tica)
                )

    ax.legend()
    ax.set_title('TICA Eigenvalues for Different Lags')
    ax.set_xticks(xvals)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')

