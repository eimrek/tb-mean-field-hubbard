# tb-mean-field-hubbard 

Python library to perform tight-binding mean field Hubbard calculations on the conjugated π-networks of organic systems.
Only carbon atoms are supported and each atom is modelled by a single p<sub>z</sub> orbital hosting a single electron.

The modelled Hamiltonian is the following:

![](https://latex.codecogs.com/svg.latex?\dpi{280}\large{\hat{H}_\text{MFH}=-t\sum\limits_{\langle{i,j}\rangle,\sigma}\left(\hat{c}^{\dag}_{i,\sigma}\hat{c}_{j,\sigma}+\text{h.c.}\right)+U\sum\limits_{i,\sigma}\langle{\hat{n}_{i,\sigma}}\rangle%20\hat{n}_{i,\overline{\sigma}}-U\sum\limits_{i}\langle{\hat{n}_{i,\uparrow}}\rangle\langle{\hat{n}_{i,\downarrow}}\rangle,})

where c<sup>†</sup>, c and n are respectively the creation, annihiliation and number operators, t is the hopping integral and U denotes the on-site Coulomb repulsion.

Example jupyter notebook `mfh.ipynb` is provided that performs the calculation for the Clar's goblet molecule.

Python dependencies:
* Standard python libraries: `numpy`, `matplotlib`
* Atomistic simulation environment: `ase`
* Python Tight Binding: `pythtb`
