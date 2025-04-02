import pandas as pd
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis import distances
from scipy.optimize import curve_fit


def calc_dmap(domain0,domain1):
    """ Distance map (nm) for single configuration

    Input: Atom groups
    Output: Distance map"""
    dmap = distances.distance_array(domain0.positions, # reference
                                    domain1.positions, # configuration
                                    box=domain0.dimensions) / 10.
    return dmap


def scaling_exp(n,r0,v):
    rh = r0 * n**v
    return rh


def fit_scaling_exp(u,ag,r0=None,traj=True,start=None,stop=None,step=None,slic=[],ij0=5):
    """ Fit scaling exponent of single chain

    Input:
      * mda Universe
      * atom group
    Output:
      * ij seq distance
      * dij cartesian distance
      * r0
      * v (scaling exponent)
      
    Author: Soeren von Buelow and Giulio Tesei
    """
    N = len(ag)
    dmap = np.zeros((N,N))
    if traj:
        if len(slic) == 0:
            for t,ts in enumerate(u.trajectory[start:stop:step]):
                m = calc_dmap(ag,ag)
                dmap += m**2 # in nm
            dmap /= len(u.trajectory[start:stop:step])
        else:
            for t,ts in enumerate(u.trajectory[slic]):
                m = calc_dmap(ag,ag)
                dmap += m**2 # in nm
            dmap /= len(u.trajectory[slic])
        dmap = np.sqrt(dmap) # RMS
    else:
        dmap = calc_dmap(ag,ag) # in nm
    ij = np.arange(N)
    dij = []
    for i in range(N):
        dij.append([])
    for i in ij:
        for j in range(i,N):
            dij[j-i].append(dmap[i,j]) # in nm

    for i in range(N):
        dij[i] = np.mean(dij[i])
    dij = np.array(dij)
    if r0 == None:
        (r0, v), pcov = curve_fit(scaling_exp,ij[ij0:],dij[ij0:])
        perr = np.sqrt(np.diag(pcov))
        verr = perr[1]
    else:
        v, pcov = curve_fit(lambda x, v: scaling_exp(x,r0,v), ij[ij0:], dij[ij0:])
        v = v[0]
        perr = np.sqrt(np.diag(pcov))
        verr = perr[0]
    return ij, dij, r0, v, verr


def weighted_average(values, errors):
    weights = 1 / np.array(errors) ** 2  # Compute weights
    weighted_avg = np.sum(weights * values) / np.sum(weights)
    weighted_error = np.sqrt(1 / np.sum(weights))  # Error on weighted mean
    return weighted_avg, weighted_error


def simple_average(values):
    mean = np.mean(values)
    error = np.std(values, ddof=1) / np.sqrt(len(values))  # Standard error
    return mean, error


if __name__=="__main__":
    head_directories = ['./-0.37','./unphosphorylated','./0.00','./charged','./phosphomimetic']
    names   = ['10pAsh1', 'pCTD2', 'pSic-6fold', 'pSic-7fold', 'HPPAGE4', 'CPPAGE4', 'SN15p', 'rOPNp', 'pERa','pTauS','pTauT']
    names_up= ['Ash1', 'CTD2', 'Sic1', 'Sic1', 'WTPAGE4', 'WTPAGE4', 'SN15', 'rOPN', 'ERa', 'TauT', 'TauS']
    indices = [names,names_up,names,names,names]
    posts   = ['_p','_up','_p0','_ch','_mim'] 


    df_analysis = pd.DataFrame(index=names,columns=['nu_p','nu_error_p','nu_up','nu_error_up','nu_p0','nu_error_p0','nu_ch','nu_error_ch','nu_mim','nu_error_mim'])
    
    for hd,proteins,post in zip(head_directories,indices,posts):
        for idx,prot in zip(names,proteins):
            nu_values=[]
            nu_errors=[]
            for replica in range(3):
                path=F"{hd}/{prot}/{replica}"
                u = Universe(f'{path:s}/top.pdb',f'{path:s}/traj.dcd',in_memory=True)
                ag = u.select_atoms("all")
                _, _, _, nu, nu_err = fit_scaling_exp(u,ag)
                nu_values.append(nu)
                nu_errors.append(nu_err)
            weighted_mean, weighted_error = weighted_average(nu_values, nu_errors)
            df_analysis.loc[idx,F'nu{post}']       = weighted_mean
            df_analysis.loc[idx,F'nu_error{post}'] = weighted_error
    df_analysis.to_csv('./nu.csv')
