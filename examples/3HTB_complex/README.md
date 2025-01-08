# Ligand-Protein Complex Tutorial
This tutorial explains how to simulate a ligand-protein complex system using the NeoBinder package with PDB code 3HTB.

## 1. Prepare System
### 1.1 Get Initial Structure
Use the `wget` command to download the initial `.pdb` file from the RCSB website.
```bash
wget https://files.rcsb.org/view/3HTB.pdb
```
### 1.2 Protein Preparation
Process the `.pdb` file to retain only the protein structure. Here, we use the `pdbfixer` package in Python to generate `3htb_pro_fix.pdb` file.

```python
import numpy as np
from openmm import unit
from openmm.vec3 import Vec3
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def fix_protein(protein_path, padding=1.0 * unit.nanometer, pH_value=7.4, addH=True):
    protein_pdb = PDBFixer(filename=protein_path)
    protein_pdb.findMissingResidues()
    protein_pdb.findMissingAtoms()
    protein_pdb.findNonstandardResidues()
    protein_pdb.replaceNonstandardResidues()
    protein_pdb.addMissingAtoms()
    if addH:
        protein_pdb.addMissingHydrogens(pH_value)
    protein_pdb.removeHeterogens(False)
    print("Residues:", protein_pdb.missingResidues)
    print("Atoms:", protein_pdb.missingAtoms)
    print("Terminals:", protein_pdb.missingTerminals)
    print("Non-standard:", protein_pdb.nonstandardResidues)

    positions = []
    for pos in protein_pdb.positions:
        positions.append(pos.value_in_unit(unit.nanometer))

    positions = np.array(positions)
    box_vec = np.eye(3) * (
        (positions.max(0) - positions.min(0)).max()
        + padding.value_in_unit(unit.nanometer) * 2
    )

    protein_pdb.topology.setPeriodicBoxVectors(box_vec)

    print("Uses Periodic box:", protein_pdb.topology.getPeriodicBoxVectors())
    return protein_pdb

input_pdb='/work_dir/sys_prep/3HTB.pdb'
output_pdb='/work_dir/sys_prep/3htb_pro_fix.pdb'
protein_pdb = fix_protein(input_pdb, 
                          padding=1 * unit.nanometer,
                          addH=False)

with open(output_pdb, 'w') as outfile:
    PDBFile.writeFile(protein_pdb.topology,
                      protein_pdb.positions,
                      file=outfile,
                      keepIds=True)
```
The generated fixed.pdb file must meet the following conditions:

- Add all missing residues and atoms.
- Include a 'CRYST1' line to describe the box of the system.
- All hydrogen atoms should be removed; they will be added automatically during preparing system.
- Only protein atoms are kept, but special water molecule or metal ions can be added at the end of the file if necessary.
### 1.3 Ligand Preparation
Extract ligand atoms from the initial `.pdb` file into `jz4.pdb`:

```bash
grep 'JZ4' ./3HTB.pdb > jz4.pdb
```
Generate an `.sdf` file from `jz4.pdb` using the `RDKit` package in Python to ensure the correct structure.
```python
from rdkit import Chem
from rdkit.Chem import rdFMCS,AllChem

def mol_smiles_to_pos_mol(mol_pos,smiles):
    mol_top=Chem.MolFromSmiles(smiles)
    mol_top = Chem.AddHs(mol_top)
    mols=[mol_pos,mol_top]
    params = rdFMCS.MCSParameters()
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareAny
    mcs = rdFMCS.FindMCS(mols, params)
    
    match_pos = mol_pos.GetSubstructMatch(mcs.queryMol)
    match_top = mol_top.GetSubstructMatch(mcs.queryMol)

    AllChem.EmbedMolecule(mol_top)
    conf = mol_top.GetConformer(0)
    for id1,id2 in zip(match_pos,match_top):
        _pos = mol_pos.GetConformer(0).GetAtomPosition(id1)
        conf.SetAtomPosition(id2, _pos)    

    mp = AllChem.MMFFGetMoleculeProperties(mol_top)
    ff = AllChem.MMFFGetMoleculeForceField(mol_top, mp)
    for i in match_top:
        ff.MMFFAddPositionConstraint(i, 0, 1.e4)
    
    ff.Minimize()
    return mol_top
    
mol = mol_smiles_to_pos_mol(Chem.MolFromPDBFile('/work_dir/sys_prep/jz4.pdb'),
                              'CCCc1ccccc1O')
Chem.MolToMolFile(mol,
                 '/work_dir/sys_prep/jz4.sdf')
```
While a `.pdb` file is also usable, an `.sdf` file is always recommended for ligands because it contains more detailed information and preserves the correct ligand topology.

If you use `jz4.pdb` in this case, the 6-member ring of the ligand may be considered as cyclohexane instead of benzene, which is incorrect.

Any method to generat `.sdf` file is allowed, just ensure the following in the `.sdf` file:
- Add all hydrogen atoms.
- Check bond types (double, triple, aromatic).
- Check chirality of carbon atoms.
- Add a 'M CHG' line for charged atoms, e.g., `-1 charged O` or `+1 charged N`.

[Clik this link for guidance of sdf file format](https://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx)
### 1.4 Run `prepare.yaml`
Edit and run the prepare.yaml file to prepare the system.

```bash
python3 /NeoDynamics/bin/prepare_openmm_system.py ./prepare.yaml 
```
Example prepare.yaml file:
```yaml
# protein information in this section
# if your system don't need protein, just delete this section
protein:
  # for most cases, only the path to fixed.pdb is enough
  path: /work_dir/sys_prep/3htb_pro_fix.pdb

# ligand information in this section
# if your system don't need protein, just delete this section
ligands:
  # 'lig1' is a user defined name to specify individual ligand in a system
  # this name won't effect any thing in output, unique is the only need
  # if you want two molecules in a system,no matter they are same or different
  # using lig1/lig2 or ligA/ligB containing individual information about 'path' 'smiles' etc
  lig1:
    # path to the ligand sdf file, with desired bond type/chiral atoms/charged atoms
    path: /work_dir/sys_prep/jz4.sdf
    # smiles of ligand with target topology, charged atoms/chiral atoms should be writen explicitly in smiles  
    smiles: 'CCCc1ccccc1O'
    # optional, set a user favorable resname
    resname: JZ4

# set other parameters for preparing system, while in most cases just keep current value
additional:
  # if or not add water and Na+/Cl-
  add_solv_ions: true
  # if or not  add H atoms
  add_hydrogens: true
  # ions concentration to netrual system charge
  ion_Strength: 0.1

# set which forcefield and water model to use
ff_setting: 
  base_ff: amber/protein.ff14SB.xml
  water_model: amber/tip3p_standard.xml

# set output path for prepared system
output_dir: /work_dir/sys_prep/3htb

```
Generated files after this step:
- system.xml: Topology information of the whole system.
- solv.pdbx: Initial structure positions.
- ligand.json: Ligand information (if applicable).
## 2.Minimize
Edit and run the `min.yaml` file to minimize the system.
```bash
python3 /NeoDynamics/bin/run_generic_md.py ./min.yaml
```
Example `min.yaml` file:
```yaml
# select a type of simulation, "min" stands for minimization
method: min
# parameters for minimization, while in most cases just keep current value
min_params:
  tolerance: 10 
  maxiter: 10000
# set a integrator
# Algorithmically, an integrator is not needed, but the current program depends on it. 
# Thus, this block must still be specified.
integrator:
  dt: 0.001
  friction_coeff: 1.0
# paths to input files
input_files:
  # coordiante file,either .pdbx or .pdb is ok, only demanding all atoms in right order
  complex: /work_dir/sys_prep/3htb/solv.pdbx
  # topology file in .xml format
  system: /work_dir/sys_prep/3htb/system.xml
  # ligand.jsong if ligand existing in system  
  ligands: /work_dir/sys_prep/3htb/ligand.json
# path for output
output:
  output_dir: /work_dir/min

```
Generated files:
- logger.log: Simulation start and end times.
- last.state: System state at the end of simulation.
- last_system.xml: Topology information of the whole system.
- last.pdbx: Positions at the end of simulation.
- last.ckpt: Checkpoint file for system state at the end of simulation.
## 3. Equilibrium
Edit and run the `eq.yaml` file to perform equilibrium simulation.
```bash
python3 /NeoDynamics/bin/run_generic_md.py ./eq.yaml
```
Example `eq.yaml` file:
```yaml
# select a type of simulation, "eq" stands for equilibrium
method: eq
# total steps for simulation to run
steps: 5000

# set a integrator
integrator:
  # time step in units of pico-second
  dt: 0.002
  # friction coefficient in units of 1/pico-second
  friction_coeff: 1.0
# set barostat to control pressure
barostat:
  # set frequency to control barostat, in this case control barostat per 25 step
  frequency: 25
  # control pressure in units of bar
  pressure: 1.0
# set temperature in units of kelvin while simulation
temperature: 298
# random seed for integrator and barostat
seed: 0

# paths to input files
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/3htb/system.xml
  ligands: /work_dir/sys_prep/3htb/ligand.json

#output settings
output:
  # set output path
  output_dir: /work_dir/eq/
  # set how many steps to output trajectory
  trajectory_interval: 1000
  # set how many steps to output checkpoint
  checkpoint_interval: 1000

```
Generated files:
- logger.log: Simulation start and end times.
- output.dcd: Positions updated every trajectory_interval steps.
- output.ckpt: Checkpoint updated every checkpoint_interval steps.
- last.state: System state at the end of simulation.
- last_system.xml: Topology information of the whole system.
- last.pdbx: Positions at the end of simulation.
- last.ckpt: Checkpoint file for system state at the end of simulation.

## 4. Equilibrium with restraints
To add additional restraint forces during equilibrium, include a `restraint` block in the `.yaml`. Here is an example showing how to restraint the center of protein at the center of the box, which can aviod cross-boundary situation:
```yaml
method: eq
steps: 5000

integrator:
  dt: 0.002
  friction_coeff: 1.0
barostat:
  frequency: 25
  pressure: 1.0
temperature: 298
seed: 0

# restraint block to add additional restraint forces
restraint:
  # user defined unique name to specify each restraint
  restr_com:
    # select a restraint type
    # dist_ref_position, restraint center of mass of a group of atoms to a reference position
    type: dist_ref_position
    # select a group of atoms to restrain,
    # selection is based on atom index,which is always 0 for the first atom then +1 along the order
    # selection of index always in string format
    # indices between atoms are separated by commas 
    restr_grp:  '4,21,35,54,74,89,106,125,149,168,180,195,202,221,245,264,286,305,326,348,360,374,389,396,417,438,452,471,478,497,504,521,540,559,573,595,605,620,632,651,665,675,685,707,718,733,752,764,786,796,815,822,846,860,871,885,892,908,927,941,963,975,990,1000,1015,1037,1056,1076,1090,1107,1119,1135,1147,1157,1167,1183,1207,1214,1233,1252,1276,1290,1300,1322,1341,1362,1377,1393,1414,1426,1437,1456,1468,1478,1494,1518,1542,1553,1563,1573,1592,1606,1623,1639,1659,1676,1693,1700,1715,1729,1736,1752,1762,1769,1789,1803,1817,1828,1847,1871,1888,1907,1924,1941,1963,1987,2011,2023,2038,2048,2058,2074,2088,2107,2117,2139,2150,2174,2198,2219,2233,2250,2263,2278,2290,2314,2324,2346,2370,2386,2405,2419,2433,2453,2477,2491,2498,2512,2536,2548,2558,2579,2601,2615'
    # set the reference position in units of nanometer, here set half size of box as center of box
    ref_position_nm: 3.46925, 3.46925, 3.46925
    # set the distance tolerance for this restraint
    # only add restraints when distance > max_nm or distance < min_nm
    max_nm: 0.1
    min_nm: 0
    # set the force constant for this restraint
    restr_k: 1000 


input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/3htb/system.xml
  ligands: /work_dir/sys_prep/3htb/ligand.json

output:
  output_dir: /work_dir/eq_restraints/
  # if or not to ducument restraint information  
  report_restraint: true
  # set how many steps to document the restraint information
  report_interval: 1000
  trajectory_interval: 1000
  checkpoint_interval: 1000
```
If `report_restraint: true` is set, a `restraint.dat` file will be generated in the output path, documenting the restraint information.