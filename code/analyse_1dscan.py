#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:21:41 2024

@author: Arrien S. Rauh
"""
import subprocess
import os
import pandas as pd
import numpy as np
import mdtraj as md
import shutil
import time


proteins = pd.read_csv('proteins.csv', index_col=0)
delta_lambdas = np.round(np.append(np.arange(-.62,0,.05),[0.0]),2)
print(delta_lambdas)
#delta_lambdas = np.round(np.arange(-.62,0,0.05),2)
# or lambdas = np.arange(0,0.3,0.05)
num_replicas = 3  # Define the number of replicas


indices = []
# or for lam in lambdas:
for delta_lambda in delta_lambdas:
    for protein in proteins.index[:]:
        for k in range(num_replicas):
            indices.append((delta_lambda,protein,k))


merged_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples=indices,
                                                         names=('dlambda',
                                                                'protein',
                                                                'replicum')),
                         columns=['exp_rg', 'exp_rg_err','exp_ree', 'exp_ree_err', 
                                  'temp', 'ionic', 'pH','seq',
                                  'rg', 'rg_err', 'bs_rg', 'acf_rg_2',
                                  'ree', 'ree_err', 'bs_ree', 'acf_ree_2'])
rg_arrays = []
ree_arrays = []

### Compile simulation results ###

# or for lam in lambdas:
for delta_lambda in delta_lambdas:
    folder = f'./{delta_lambda:.2f}'
    os.makedirs(folder, exist_ok=True)
    residues = pd.read_csv('residues.csv', index_col=0)
    residues.loc['B', 'lambdas'] -= delta_lambda
    residues.loc['O', 'lambdas'] -= delta_lambda
    # or residues.loc['B', 'lambdas'] = lam
    #    residues.loc['O', 'lambdas'] = lam
#    residues.to_csv(folder+'/residues.csv')

    for protein in proteins.index[:]:

        for k in range(num_replicas):
            subfolder = folder+F'/{protein:s}/{k:d}'
            print(subfolder)

            if not os.path.isdir(subfolder):
                pass

            if os.path.isfile(F"{subfolder}/analysis.csv") and os.path.isfile(F"{subfolder}/rg_array.npy"):
                df_temp  = pd.read_csv(F"{subfolder}/analysis.csv",index_col=0)
                rg_array = np.load(F"{subfolder}/rg_array.npy")
                merged_df.loc[(delta_lambda,protein,k)] = df_temp.loc[protein]
                rg_arrays.append(rg_array)
                ree_array = np.load(F"{subfolder}/ree_array.npy")
                ree_arrays.append(ree_array)

# print(len(rg_arrays),merged_df.shape)
print(merged_df.index)
merged_df['rg_array'] = rg_arrays
merged_df['ree_array'] = rg_arrays


### Merging replicas ###
indices = []
# or for lam in lambdas:
for delta_lambda in delta_lambdas:
    for protein in proteins.index[:]:
        indices.append((delta_lambda,protein))


df_protein = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples=indices,
                                                          names=('dlambda',
                                                                 'protein')),
                          columns=['exp_rg', 'exp_rg_err','exp_ree', 'exp_ree_err',
                                   'temp', 'ionic', 'pH', 'seq',
                                   'rg', 'rg_err', 'ree', 'ree_err',
                                   'bs_rg0','bs_rg1','bs_rg2',
                                   'bs_ree0','bs_ree1','bs_ree2',
                                   'acf_rg0','acf_rg1','acf_rg2',
                                   'acf_ree0','acf_ree1','acf_ree2'])


# or for lam in lambdas:
for delta_lambda in delta_lambdas:
    for protein in proteins.index[:]:
        for col in ['exp_rg', 'exp_rg_err','exp_ree', 'exp_ree_err', 'temp', 'ionic', 'pH','seq']:
            df_protein.loc[(delta_lambda,protein),col] = merged_df.loc[(delta_lambda,protein,0),col]
        
        df_protein.loc[(delta_lambda,protein),'rg']      = np.mean([merged_df.loc[(delta_lambda,protein,i),'rg'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'rg_err']  = np.std([merged_df.loc[(delta_lambda,protein,i),'rg'] for i in [0,1,2]])/np.sqrt(3)
        df_protein.loc[(delta_lambda,protein),'ree']     = np.mean([merged_df.loc[(delta_lambda,protein,i),'ree'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'ree_err'] = np.std([merged_df.loc[(delta_lambda,protein,i),'ree'] for i in [0,1,2]])/np.sqrt(3)

        df_protein.loc[(delta_lambda,protein),'bs_rg0'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'bs_rg'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'bs_rg1'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'bs_rg'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'bs_rg2'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'bs_rg'] for i in [0,1,2]])

        df_protein.loc[(delta_lambda,protein),'bs_ree0'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'bs_ree'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'bs_ree1'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'bs_ree'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'bs_ree2'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'bs_ree'] for i in [0,1,2]])

        df_protein.loc[(delta_lambda,protein),'acf_rg0'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'acf_rg_2'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'acf_rg1'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'acf_rg_2'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'acf_rg2'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'acf_rg_2'] for i in [0,1,2]])

        df_protein.loc[(delta_lambda,protein),'acf_ree0'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'acf_ree_2'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'acf_ree1'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'acf_ree_2'] for i in [0,1,2]])
        df_protein.loc[(delta_lambda,protein),'acf_ree2'] = np.mean([merged_df.loc[(delta_lambda,protein,i),'acf_ree_2'] for i in [0,1,2]])

df_protein['rel_err_rg'] = df_protein.apply(lambda x: (x["rg"]-x["exp_rg"])/x["exp_rg"],axis=1)
df_protein['rel_err_ree'] = df_protein.apply(lambda x: (x["ree"]-x["exp_ree"])/x["exp_ree"],axis=1)


merged_df.to_pickle("merged_df.pkl")
merged_df[merged_df.columns[:-1]].to_csv("merged_df.csv",sep=';')
df_protein.to_pickle("df_protein.pkl")#,sep=';')