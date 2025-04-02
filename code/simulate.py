import mdtraj as md
import pandas as pd
import numpy as np
import sys, os
if os.path.exists('/groups/sbinlab/asrauh/software/BLOCKING'):
    sys.path.append('/groups/sbinlab/asrauh/software/BLOCKING')
else:
    sys.path.append('./BLOCKING')
from main import BlockAnalysis
from statsmodels.tsa.stattools import acf
from openmm.app import *
from openmm import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',dest='name',type=str)
parser.add_argument('--folder',dest='folder',type=str)
parser.add_argument('--dlambda',dest='dlambda',type=float)
args = parser.parse_args()


def xy_spiral_array(n, delta=0, arc=.38, separation=.7):
    """
    create points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    """
    def p2c(r, phi):
        """
        polar to cartesian
        """
        return (r * np.cos(phi), r * np.sin(phi))
    r = arc
    b = separation / (2 * np.pi)
    phi = float(r) / b
    coords = []
    for i in range(n):
        coords.append(list(p2c(r, phi))+[0])
        phi += float(arc) / r
        r = b * phi
    return np.array(coords)+delta


def genParamsDH(residues,seq,pH,ionic,temp):
    kT = 8.3145*temp*1e-3
    fasta = list(seq)
    r = residues.copy()
    r.q = r.q.astype(float)
    # Set the charge of His based on the pH of the protein solution
    r.loc['H','q'] = 1/(1+10**(pH-r.loc['H','pKa']))
    # Set the charge of pTyr, pSer and pThr based on the pH of the protein solution
    r.loc['B','q'] = -(1+1/(1+10**(r.loc['B','pKa']-pH))) # pSER
    r.loc['O','q'] = -(1+1/(1+10**(r.loc['O','pKa']-pH))) # pThr
    r.loc['U','q'] = -(1+1/(1+10**(r.loc['U','pKa']-pH))) # pTyr
    # Set the charge of termini
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['X','q'] = r.loc[seq[0],'q'] + 1.
    fasta[0] = 'X'
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['Z','q'] = r.loc[seq[-1],'q'] - 1.
    fasta[-1] = 'Z'
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    return yukawa_eps, yukawa_kappa


def genXTC(name, eq_steps=10):
     """
     Generates coordinate and trajectory
     in convenient formats
     """
     traj = md.load_dcd(f'{name:s}/traj.dcd', top=f'{name:s}/top.pdb')
     cgtop = md.Topology()
     cgchain = cgtop.add_chain()
     for atom in traj.top.atoms:
         cgres = cgtop.add_residue(atom.name, cgchain)
         cgtop.add_atom(atom.name, element=md.element.carbon, residue=cgres)
     for i in range(traj.n_atoms-1):
         cgtop.add_bond(cgtop.atom(i),cgtop.atom(i+1))
     traj = md.Trajectory(traj.xyz, cgtop, traj.time, traj.unitcell_lengths, traj.unitcell_angles)
     traj.image_molecules(inplace=True, anchor_molecules=[set(traj.top.atoms)], make_whole=True)
     traj.center_coordinates()
     traj.xyz += traj.unitcell_lengths[0,0]/2
     traj[int(eq_steps):].save_xtc(f'{name:s}/traj_eq.xtc')
     traj[int(eq_steps)].save_pdb(f'{name:s}/top_eq.pdb')


def autoblock(cv, multi=1):
    block = BlockAnalysis(cv, multi=multi)
    block.SEM()
    return block.av, block.sem, block.bs


def analyse(residues,proteins,name,folder):
    traj = md.load_dcd(f'{folder:s}/traj.dcd', top=f'{folder:s}/top.pdb')
    ### Rg ###
    masses = residues.loc[list(proteins.loc[name].seq),'MW'].values
    masses[0] += 2
    masses[-1] += 16
    # calculate the center of mass
    cm = np.sum(traj.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate residue-cm distances
    si = np.linalg.norm(traj.xyz - cm[:,np.newaxis,:],axis=2)
    # calculate rg
    rg_array = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
    _, rg_se, rg_blocksize = autoblock(rg_array)
    # calculate acf
    acf_rg_2 = acf(rg_array,nlags=2,fft=True)[2]

    ### ete ###
    # calculate ete
    ete_array = np.linalg.norm(traj.xyz[:,0]-traj.xyz[:,-1],axis=1)
    ete_m = np.mean(ete_array)
    _, ete_se, ete_blocksize = autoblock(ete_array)
    # calculate acf
    acf_ete_2 = acf(ete_array,nlags=2,fft=True)[2]
    return rg_array, rg_se, rg_blocksize, acf_rg_2, ete_array, ete_se, ete_blocksize, acf_ete_2



def simulate(residues,proteins,name,folder):

    temp = proteins.loc[name].temp
    ionic = proteins.loc[name].ionic
    pH = proteins.loc[name].pH
    seq = proteins.loc[name].seq

    yukawa_eps, yukawa_kappa = genParamsDH(residues,seq,pH,ionic,temp)

    N_res = len(seq)
    L = (N_res-1)*0.38+4
    if L<15:
        L=15
    N_save = 7000 if N_res < 150 else int(np.ceil(3e-4*N_res**2)*1000)
    N_steps = 5050*N_save

    system = openmm.System()

    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = L * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # initial configuration
    top = md.Topology()
    pos = []
    chain = top.add_chain()
    #pos.append([[0,0,L/2+(i-N_res/2.)*.38] for i in range(N_res)])
    pos.append(xy_spiral_array(n=N_res,delta=L/2.))
    for resname in seq:
        residue = top.add_residue(resname, chain)
        top.add_atom(resname, element=md.element.carbon, residue=residue)
    for i in range(chain.n_atoms-1):
        top.add_bond(chain.atom(i),chain.atom(i+1))
    md.Trajectory(np.array(pos).reshape(N_res,3), top, 0, [L,L,L], [90,90,90]).save_pdb(f'{folder:s}/top.pdb')

    pdb = app.pdbfile.PDBFile(f'{folder:s}/top.pdb')

    system.addParticle((residues.loc[seq[0]].MW+2)*unit.amu)
    for a in seq[1:-1]:
        system.addParticle(residues.loc[a].MW*unit.amu)
    system.addParticle((residues.loc[seq[-1]].MW+16)*unit.amu)

    hb = openmm.HarmonicBondForce()
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/2)^12-(0.5*(s1+s2)/2)^6')
    yu = openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r - exp(-kappa*4)/4); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addPerParticleParameter('q')

    ah.addGlobalParameter('eps',0.2*4.184*unit.kilojoules_per_mole)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    for a,e in zip(seq,yukawa_eps):
        yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
        ah.addParticle([residues.loc[a].sigmas*unit.nanometer, residues.loc[a].lambdas*unit.dimensionless])

    for i in range(N_res-1):
        hb.addBond(i, i+1, 0.38*unit.nanometer, 8033*unit.kilojoules_per_mole/(unit.nanometer**2))
        yu.addExclusion(i, i+1)
        ah.addExclusion(i, i+1)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(2*unit.nanometer)

    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    platform = openmm.Platform.getPlatformByName('CPU')
    check_point = f'{folder:s}/restart.chk'

    if not os.path.isfile(check_point):
        integrator = openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.005*unit.picosecond) # 5 fs
        simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(Threads='1'))
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.step(100000)
        integrator = openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.01*unit.picosecond) # 10 fs
        simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(Threads='1'))
        simulation.context.setPositions(pdb.positions)
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{folder:s}/traj.dcd',int(N_save)))
    else:
        print('Reading check point file')
        integrator = openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.01*unit.picosecond) # 10 fs
        simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(Threads='1'))
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{folder:s}/traj.dcd',int(N_save),append=True))

    simulation.reporters.append(app.statedatareporter.StateDataReporter(file=f'{folder:s}/log',reportInterval=int(N_save*10),totalSteps=N_steps,progress=True,step=True,speed=True,elapsedTime=True,separator='\t',append=True))

    simulation.step(N_steps)



folder = args.folder
proteins = pd.read_csv('./proteins.csv',index_col=0)
#residues = pd.read_csv(folder.split('_')[0]+'/residues.csv',index_col=0)
residues = pd.read_csv(F'./{args.dlambda}/residues.csv',index_col=0)

simulate(residues,proteins,args.name,folder)

###
df_analysis = proteins.loc[[args.name]].copy()
# Calculate Rg and Ree values and SEM
rg_array, rg_se, rg_blocksize, acf_rg_2, ete_array, ete_se, ete_blocksize, acf_ete_2 = analyse(residues,proteins,args.name,folder)

# Save arrays
np.save(f'{folder:s}/rg_array.npy', rg_array)
np.save(f'{folder:s}/ree_array.npy', ete_array)

# Store data in DataFrame and save it
df_analysis.loc[args.name,'rg'] = np.mean(rg_array)
df_analysis.loc[args.name,'rg_err'] = rg_se
df_analysis.loc[args.name,'bs_rg'] = rg_blocksize
df_analysis.loc[args.name,'acf_rg_2'] = acf_rg_2
df_analysis.loc[args.name,'ree'] = np.mean(ete_array)
df_analysis.loc[args.name,'ree_err'] = ete_se
df_analysis.loc[args.name,'bs_ree'] = ete_blocksize
df_analysis.loc[args.name,'acf_ree_2'] = acf_ete_2
df_analysis.to_csv(F'{folder:s}/analysis.csv')
