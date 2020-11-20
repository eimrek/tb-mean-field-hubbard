# tb-mean-field-hubbard 

Python library to perform tight-binding mean field Hubbard calculations on the conjugated π-networks of organic systems.
Only carbon atoms are supported and each atom is assumed a single p<sub>z</sub> orbital with a single electron.

The modelled Hamiltonian is the following:

![](https://latex.codecogs.com/svg.latex?\dpi{300}\large\hat{H}_\text{MFH}=-t\sum\limits_{\langle{i,j}\rangle,\sigma}\hat{c}^{\dag}_{i,\sigma}\hat{c}_{j,\sigma}+U\sum\limits_{i,\sigma}\langle{\hat{n}_{i,\sigma}}\rangle%20\hat{n}_{i,\overline{\sigma}}-U\sum\limits_{i}\langle{\hat{n}_{i,\uparrow}}\rangle\langle{\hat{n}_{i,\downarrow}}\rangle) 


Example jupyter notebook `mfh.ipynb` is provided that performs the calculation for the Clar's goblet molecule.

Python dependencies:
* Standard python libraries: `numpy`, `matplotlib`
* Atomistic simulation environment: `ase`
* Python Tight Binding: `pythtb`
