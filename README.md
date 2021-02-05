# tb-mean-field-hubbard 

Python library to perform tight-binding mean field Hubbard calculations on the conjugated π-networks of organic systems.
Only carbon atoms are supported and each atom is modelled by a single p<sub>z</sub> orbital hosting a single electron.

The modelled Hamiltonian is the following:

![](https://latex.codecogs.com/svg.latex?\dpi{280}\large{\hat{H}_\text{MFH}=-t\sum\limits_{\langle{i,j}\rangle,\sigma}\left(\hat{c}^{\dag}_{i,\sigma}\hat{c}_{j,\sigma}+\text{h.c.}\right)+U\sum\limits_{i,\sigma}\langle{\hat{n}_{i,\sigma}}\rangle%20\hat{n}_{i,\overline{\sigma}}-U\sum\limits_{i}\langle{\hat{n}_{i,\uparrow}}\rangle\langle{\hat{n}_{i,\downarrow}}\rangle,})

where c<sup>†</sup>, c and n are respectively the creation, annihiliation and number operators, t is the hopping integral and U denotes the on-site Coulomb repulsion.


Python dependencies:
* Standard python libraries: `numpy`, `matplotlib`
* Atomistic simulation environment: `ase`
* Python Tight Binding: `pythtb`

### Installation

Option 1) To install the dependencies and the library, one can use
```
pip install git+https://github.com/eimrek/tb-mean-field-hubbard.git#egg=tb-mean-field-hubbard
```

Option 2) To also have access to the code and the notebook, it's better to call instead
```
git clone https://github.com/eimrek/tb-mean-field-hubbard.git
cd tb-mean-field-hubbard
pip install -e .
```

Option 3) If dependencies are already installed, then simply downloading the code and executing the notebook will work.

### Example usage

Example jupyter notebook `mfh.ipynb` is provided that performs the calculation for the Clar's goblet molecule. The geometry is read from a `xyz` file.

The following image demonstrates a selection of the output for the calculation for parameters `t=2.7` and `U=3.0` (both in electronvolts).

<p align="center"><img class="marginauto" src="res/example-output.png" width="700"></p>

