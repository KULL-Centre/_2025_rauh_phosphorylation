import sys
import subprocess
import os
import pandas as pd
import numpy as np
import mdtraj as md
import shutil
from scipy.stats import pearsonr, spearmanr


sys.path.append("/projects/prism/people/mhz916/software/BLOCKING")
from main import BlockAnalysis

proteins      = pd.read_csv('./proteins.csv', index_col=0)
proteins_up   = pd.read_csv('./unphosphorylated/proteins_un.csv', index_col=0)
proteins_ch   = pd.read_csv('./charged/proteins.csv', index_col=0)
proteins_mim  = pd.read_csv('./phosphomimetic/proteins_mim.csv', index_col=0)
proteins_ppos = pd.read_csv('./check_ppos37/proteins_psic6fold.csv', index_col=0)


df_residues_37 = pd.read_csv('./-0.37/residues.csv').set_index('one')
df_residues_37.loc['B','q']=-2
df_residues_37.loc['O','q']=-2
df_residues_00 = pd.read_csv('./0.00/residues.csv').set_index('one')
df_residues_00.loc['B','q']=-2
df_residues_00.loc['O','q']=-2
df_residues_up = pd.read_csv('./unphosphorylated/residues.csv').set_index('one')
df_residues_ch = pd.read_csv('./charged/residues_ch.csv').set_index('one')


delta_lambdas = np.append(np.arange(-.62,0,.05),[0.0]) #np.arange(-.62,0,.05)
# or lambdas = np.arange(0,0.3,0.05)
num_replicas = 3  # Define the number of replicas


def delta_rg(rg_wt,rg_m,rg_wt_err,rg_m_err):
    dRg = rg_m.astype(float)-rg_wt.astype(float)
    dRg_err = np.sqrt(np.power(np.power(rg_m_err.astype(float),2)+rg_wt_err.astype(float),2))
    return dRg, dRg_err


def propagate_error_sum(err1,err2,round=5):
    return np.round(np.sqrt(err1**2+err2**2),round)


def propagate_error_ratio(val1,val2,err1,err2,round=5):
    return np.round(np.sqrt(np.power(err1/val1,2)**2+np.power(err2/val2,2)),round)


### Calculating performance per parameter ###
def chi_squared(exp,exp_err,sim):
    return np.sum((exp-sim)**2/(exp_err)**2)


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between two datasets.

    Parameters:
    y_true (list or np.array): Actual values
    y_pred (list or np.array): Predicted values

    Returns:
    float: RMSE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_target(protein_names, data, target='rg'):
    indices = []
    # or for lam in lambdas:
    for delta_lambda in delta_lambdas:
        delta_lambda = np.round(delta_lambda,2)
        indices.append(delta_lambda)

    df_par = pd.DataFrame(index=indices,
                          columns=[F'chi2',
                                   F'chi2_ne',
                                   F'mre',
                                   F'mse',
                                   F'rmse',
                                   F'mae',
                                   *[i+'_c2' for i in protein_names],
                                   *[i+'_c2_ne' for i in protein_names],
                                   *[i+'_re' for i in protein_names]])
    for delta_lambda in delta_lambdas:
        delta_lambda = np.round(delta_lambda,2)
        ## Relative error
        temp_rel = []
        for protein,col in zip(protein_names,[i+'_re' for i in protein_names]):
            df_par.loc[delta_lambda,col] = data.loc[(delta_lambda,protein),F'rel_err_{target}']
            temp_rel.append(data.loc[delta_lambda,F'rel_err_{target}'])
        ## Chi-squared w/o error
        temp_chi2_ne = []
        for protein,col in zip(protein_names,[i+'_c2_ne' for i in protein_names]):
            df_par.loc[delta_lambda,col] = data.loc[(delta_lambda,protein),F'chi2_{target}_ne']
            temp_chi2_ne.append(data.loc[(delta_lambda,protein),F'chi2_{target}_ne'])
        ## Chi-squared w error
        temp_chi2 = []
        for protein,col in zip(protein_names,[i+'_c2' for i in protein_names]):
            df_par.loc[delta_lambda,col] = data.loc[(delta_lambda,protein),F'chi2_{target}']
            temp_chi2.append(data.loc[(delta_lambda,protein),F'chi2_{target}'])
        ## Overall metrics       
        df_par.loc[delta_lambda,'mre']     = np.mean(np.abs(temp_rel))
        df_par.loc[delta_lambda,'chi2_ne'] = np.sum(temp_chi2_ne)
        df_par.loc[delta_lambda,'chi2']    = np.sum(temp_chi2)
        df_par.loc[delta_lambda,'mae']     = np.mean(np.abs(np.sqrt(temp_chi2_ne)))
        df_par.loc[delta_lambda,'mse']     = np.mean(temp_chi2_ne)
        df_par.loc[delta_lambda,'rmse']    = np.sqrt(df_par.loc[delta_lambda,'mse'])                                                            
    return df_par


PtoWT = {'10pAsh1':'Ash1','pCTD2':'CTD2','pSic-6fold':'Sic1','pSic-7fold':'Sic1',
         'HPPAGE4':'WTPAGE4','CPPAGE4':'WTPAGE4','SN15p':'SN15','rOPNp':'rOPN',
         'pERa':'ERa','pTauT':'TauT','pTauS':'TauS'}

WTtoP = {'Ash1':'10pAsh1','CTD2':'pCTD2','Sic1':'pSic-6fold','Sic1':'pSic-7fold',
         'WTPAGE4':'HPPAGE4','WTPAGE4':'CPPAGE4','SN15':'SN15p','rOPN':'rOPNp',
         'ERa':'pERa','TauT':'pTauT','TauS':'pTauS'}

# up_labels=

ph_labels= {'10pAsh1':'pAsh1','pCTD2':'pCTD2','pSic-6fold':'pSic1 (6p)','pSic-7fold':'pSic (7p)',
           'HPPAGE4':'HPPAGE4','CPPAGE4':'CPPAGE4','SN15p':'SN15p','rOPNp':'rOPNp',
           'pERa':'pERa','pTauT':'pTauT','pTauS':'pTauS'}


up_folder = "./unphosphorylated/"

for val,err in zip(['rg', 'ree', 'bs_rg', 'acf_rg_2','bs_ree', 'acf_ree_2'],['rg_err','ree_err', 'bs_rg_err', 'acf_rg_2_err', 'bs_ree_err', 'acf_ree_2_err']):
    vals_temp = []
    errs_temp = []
    for up_prot in proteins_up.index:
        reps = []
        for k in range(num_replicas):
            subfolder = up_folder+F'/{up_prot:s}/{k:d}'
            if not os.path.isdir(subfolder):
                pass
            if os.path.isfile(F"{subfolder}/analysis.csv"):
                df_temp  = pd.read_csv(F"{subfolder}/analysis.csv",index_col=0)
                reps.append(df_temp.loc[up_prot,val])
                # print(reps)
        vals_temp.append(np.mean(reps))
        errs_temp.append(np.std(reps)/np.sqrt(3))
        # print(vals_temp)
    proteins_up[val] = vals_temp
    proteins_up[err] = errs_temp    

charged_folder = "./charged/"

for val,err in zip(['rg', 'ree', 'bs_rg', 'acf_rg_2','bs_ree', 'acf_ree_2'],['rg_err','ree_err', 'bs_rg_err', 'acf_rg_2_err', 'bs_ree_err', 'acf_ree_2_err']):
    vals_temp = []
    errs_temp = []
    for ch_prot in proteins_ch.index:
        reps = []
        for k in range(num_replicas):
            subfolder = charged_folder+F'/{ch_prot:s}/{k:d}'
            if not os.path.isdir(subfolder):
                pass
            if os.path.isfile(F"{subfolder}/analysis.csv"):
                df_temp  = pd.read_csv(F"{subfolder}/analysis.csv",index_col=0)
                reps.append(df_temp.loc[ch_prot,val])
                # print(reps)
        vals_temp.append(np.mean(reps))
        errs_temp.append(np.std(reps)/np.sqrt(3))
    proteins_ch[val] = vals_temp
    proteins_ch[err] = errs_temp


mim_folder = "./phosphomimetic/"

for val,err in zip(['rg', 'ree', 'bs_rg', 'acf_rg_2','bs_ree', 'acf_ree_2'],['rg_err','ree_err', 'bs_rg_err', 'acf_rg_2_err', 'bs_ree_err', 'acf_ree_2_err']):
    vals_temp = []
    errs_temp = []
    for mim_prot in proteins_mim.index:
        reps = []
        for k in range(num_replicas):
            subfolder = mim_folder+F'/{mim_prot:s}/{k:d}'
            if not os.path.isdir(subfolder):
                pass
            if os.path.isfile(F"{subfolder}/analysis.csv"):
                df_temp  = pd.read_csv(F"{subfolder}/analysis.csv",index_col=0)
                reps.append(df_temp.loc[mim_prot,val])
                # print(reps)
        vals_temp.append(np.mean(reps))
        errs_temp.append(np.std(reps)/np.sqrt(3))
    proteins_mim[val] = vals_temp
    proteins_mim[err] = errs_temp


ppos_folder = "./check_ppos/"

for val,err in zip(['rg', 'ree', 'bs_rg', 'acf_rg_2','bs_ree', 'acf_ree_2'],['rg_err','ree_err', 'bs_rg_err', 'acf_rg_2_err', 'bs_ree_err', 'acf_ree_2_err']):
    vals_temp = []
    errs_temp = []
    for ppos_prot in proteins_ppos.index:
        reps = []
        for k in range(num_replicas):
            subfolder = ppos_folder+F'/{ppos_prot:s}/{k:d}'
            if not os.path.isdir(subfolder):
                pass
            if os.path.isfile(F"{subfolder}/analysis.csv"):
                df_temp  = pd.read_csv(F"{subfolder}/analysis.csv",index_col=0)
                reps.append(df_temp.loc[ppos_prot,val])
                # print(reps)
        vals_temp.append(np.mean(reps))
        errs_temp.append(np.std(reps)/np.sqrt(3))
    proteins_ppos[val] = vals_temp
    proteins_ppos[err] = errs_temp


prots_rg = proteins.index[:-2]
prots_rg_new = ['10pAsh1', 'pCTD2', 'pSic-6fold', 'pSic-7fold', 'HPPAGE4', 'SN15p', 'rOPNp', 'pERa']
prots_ree = proteins.index[-2:]
indices_rg  = []
indices_ree = []

for dl in np.round(delta_lambdas,2):
    for p in proteins.index[:-2]:
        indices_rg.append((dl,p))
    for p in proteins.index[-2:]:
        indices_ree.append((dl,p))

df_protein     = pd.read_pickle("./df_protein.pkl").loc[indices_rg]
df_protein_ree = pd.read_pickle("./df_protein.pkl").loc[indices_ree]

df_protein['up_rg']         = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[:-2]]],'rg'].to_numpy()
df_protein['up_rg_err']     = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[:-2]]],'rg_err'].to_numpy()
df_protein['up_exp_rg']     = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[:-2]]],'exp_rg'].to_numpy()
df_protein['up_exp_rg_err'] = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[:-2]]],'exp_rg_err'].to_numpy()

#### Rgs ####
drgs =  delta_rg(rg_wt=df_protein['up_rg'].to_numpy(),
                 rg_m=df_protein['rg'].to_numpy(),
                 rg_wt_err=df_protein['up_rg_err'].to_numpy(),
                 rg_m_err=df_protein['rg_err'].to_numpy())
exp_drgs = delta_rg(rg_wt=df_protein['up_exp_rg'].to_numpy(),
                    rg_m=df_protein['exp_rg'].to_numpy(),
                    rg_wt_err=df_protein['up_exp_rg_err'].to_numpy(),
                    rg_m_err=df_protein['exp_rg_err'].to_numpy())

df_protein['drg']         = drgs[0]
df_protein['drg_err']     = drgs[1]
df_protein['exp_drg']     = exp_drgs[0]
df_protein['exp_drg_err'] = exp_drgs[1]


df_protein['ratio_rg']      = df_protein.apply(lambda x: x["drg"]/x["rg"],axis=1)
df_protein['ratio_rg_err']  = df_protein.apply(lambda x: propagate_error_ratio(x["drg"],x["rg"],x["drg_err"],x["rg_err"]),axis=1)
df_protein['exp_ratio_rg']      = df_protein.apply(lambda x: x["exp_drg"]/x["exp_rg"],axis=1)
df_protein['exp_ratio_rg_err']  = df_protein.apply(lambda x: propagate_error_ratio(x["exp_drg"],x["exp_rg"],x["exp_drg_err"],x["exp_rg_err"]),axis=1)


df_protein['rel_err_rg']  = df_protein.apply(lambda x: (x["rg"]-x["exp_rg"])/x["exp_rg"],axis=1)
df_protein['rel_err_drg'] = df_protein.apply(lambda x: (x["drg"]-x["exp_drg"])/x["exp_drg"],axis=1)
df_protein['rel_err_ratio'] = df_protein.apply(lambda x: (x["ratio_rg"]-x["exp_ratio_rg"])/np.abs(x["exp_ratio_rg"]),axis=1)


df_protein['chi2_rg']  = df_protein.apply(lambda x: (x["exp_rg"]-x["rg"])**2/(x["exp_rg_err"])**2,axis=1)
df_protein['chi2_drg'] = df_protein.apply(lambda x: (x["exp_drg"]-x["drg"])**2/(x["exp_drg_err"])**2,axis=1)
df_protein['chi2_ratio'] = df_protein.apply(lambda x: (x["exp_ratio_rg"]-x["ratio_rg"])**2/(x["exp_ratio_rg_err"])**2,axis=1)


df_protein['chi2_rg_ne']  = df_protein.apply(lambda x: (x["exp_rg"]-x["rg"])**2,axis=1)
df_protein['chi2_drg_ne'] = df_protein.apply(lambda x: (x["exp_drg"]-x["drg"])**2,axis=1)
df_protein['chi2_ratio_ne'] = df_protein.apply(lambda x: (x["exp_ratio_rg"]-x["ratio_rg"])**2,axis=1)

#### Rees ####
df_protein_ree['up_ree']         = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[-2:]]],'ree'].to_numpy()
df_protein_ree['up_ree_err']     = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[-2:]]],'ree_err'].to_numpy()
df_protein_ree['up_exp_ree']     = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[-2:]]],'exp_ree'].to_numpy()
df_protein_ree['up_exp_ree_err'] = proteins_up.loc[[PtoWT[idx] for idx in len(delta_lambdas)*[*proteins.index[-2:]]],'exp_ree_err'].to_numpy()


drees =  delta_rg(rg_wt=df_protein_ree['up_ree'].to_numpy(),
                  rg_m=df_protein_ree['ree'].to_numpy(),
                  rg_wt_err=df_protein_ree['up_ree_err'].to_numpy(),
                  rg_m_err=df_protein_ree['ree_err'].to_numpy())
exp_drees = delta_rg(rg_wt=df_protein_ree['up_exp_ree'].to_numpy(),
                     rg_m=df_protein_ree['exp_ree'].to_numpy(),
                     rg_wt_err=df_protein_ree['up_exp_ree_err'].to_numpy(),
                     rg_m_err=df_protein_ree['exp_ree_err'].to_numpy())

df_protein_ree['dree']         = drees[0]
df_protein_ree['dree_err']     = drees[1]
df_protein_ree['exp_dree']     = exp_drees[0]
df_protein_ree['exp_dree_err'] = exp_drees[1]


df_protein_ree['ratio_ree']      = df_protein_ree.apply(lambda x: x["dree"]/x["ree"],axis=1)
df_protein_ree['ratio_ree_err']  = df_protein_ree.apply(lambda x: propagate_error_ratio(x["dree"],x["ree"],x["dree_err"],x["ree_err"]),axis=1)
df_protein_ree['exp_ratio_ree']      = df_protein_ree.apply(lambda x: x["exp_dree"]/x["exp_ree"],axis=1)
df_protein_ree['exp_ratio_ree_err']  = df_protein_ree.apply(lambda x: propagate_error_ratio(x["exp_dree"],x["exp_ree"],x["exp_dree_err"],x["exp_ree_err"]),axis=1)


df_protein_ree['rel_err_ree']   = df_protein_ree.apply(lambda x: (x["ree"]-x["exp_ree"])/x["exp_ree"],axis=1)
df_protein_ree['rel_err_dree']  = df_protein_ree.apply(lambda x: (x["dree"]-x["exp_dree"])/x["exp_dree"],axis=1)
df_protein_ree['rel_err_ratio'] = df_protein_ree.apply(lambda x: (x["ratio_ree"]-x["exp_ratio_ree"])/np.abs(x["exp_ratio_ree"]),axis=1)
#df_protein_ree.apply(lambda x: (x["ratio_ree"]-x["exp_ratio_ree"])/x["exp_ratio_ree"],axis=1)


df_protein_ree['chi2_ree']  = df_protein_ree.apply(lambda x: (x["exp_ree"]-x["ree"])**2/(x["exp_ree_err"])**2,axis=1)
df_protein_ree['chi2_dree'] = df_protein_ree.apply(lambda x: (x["exp_dree"]-x["dree"])**2/(x["exp_dree_err"])**2,axis=1)
df_protein_ree['chi2_ratio'] = df_protein_ree.apply(lambda x: (x["exp_ratio_ree"]-x["ratio_ree"])**2/(x["exp_ratio_ree_err"])**2,axis=1)


df_protein_ree['chi2_ree_ne']  = df_protein_ree.apply(lambda x: (x["exp_ree"]-x["ree"])**2,axis=1)
df_protein_ree['chi2_dree_ne'] = df_protein_ree.apply(lambda x: (x["exp_dree"]-x["dree"])**2,axis=1)
df_protein_ree['chi2_ratio_ne'] = df_protein_ree.apply(lambda x: (x["exp_ratio_ree"]-x["ratio_ree"])**2,axis=1)


proteins_up.to_csv('./processed_data/proteins_unph.csv')
proteins_ch.to_csv('./processed_data/proteins_ch.csv')
proteins_mim.to_csv('./processed_data/proteins_mim.csv')
proteins_ppos.to_csv('./processed_data/proteins_ppos.csv')
df_protein.to_csv('./processed_data/df_protein.csv')
df_protein_ree.to_csv('./processed_data/df_protein_ree.csv')

