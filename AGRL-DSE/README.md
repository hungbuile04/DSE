# AGRL-DSE
> Code and Datasets for "Adaptive graph representation learning on a heterogeneous graph for drug side effects prediction" https://github.com/tanhe16/AGRL-DSE.
## Datasets
- data/SIDER-frequency.csv is the is drug side effect association data, which contains 37,441 frequency classes, including 759 drugs and 994 adverse effects. 

- data/SIDER-SMILES.csv is the drug SMILES data, which contains the Simplified Molecular Input Line Entry System (SMILES) representations for 759 drugs.

- data/SIDER-side_effect.csv is the side effect descriptor data, which includes detailed descriptors for 994 adverse effects.

- data/FAERS-correlation.csv is the is drug side effect association data, which included 1,073 drugs, 893 side effects, and 141,752 interaction associations. 

- data/FAERS-SMILES.csv is the drug SMILES data, which contains the Simplified Molecular Input Line Entry System (SMILES) representations for 1073 drugs.

- data/FAERS-ADR.csv is the side effect descriptor data, which includes detailed descriptors for 893 adverse effects.
## Code
### Environment Requirement
The code has been tested running under Python 3.7. The required packages are as follows:
- numpy == 1.15.4
- scipy == 1.1.0
- tensorflow == 1.12.0
### Usage
```shell
git clone https://github.com/tanhe16/AGRL-DSE.git.
python main.py
```
