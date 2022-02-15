import numpy as np
import scipy.special

import matplotlib.pyplot as plt

import igor_tools

### ------------------------------------------------------------------------------
### PUBLIC TOOLS
### ------------------------------------------------------------------------------

def create_itx(data, extent, wavename, filename):
    dx = (extent[1] - extent[0]) / data.shape[0]
    dy = (extent[3] - extent[2]) / data.shape[1]
    xaxis = (extent[0], dx, "[ang]")
    yaxis = (extent[2], dy, "[ang]")
    igor_tools.write_2d_itx(filename, data, xaxis, yaxis, wavename)

### ------------------------------------------------------------------------------
### GEOMETRY TOOLS
### ------------------------------------------------------------------------------

def scale_atoms(atoms, factor):
    cell_defined = True
    if atoms.cell is None or (atoms.cell == 0.0).all():
        cell_defined = False
        atoms.cell = np.diag(np.ptp(atoms.positions, axis=0))
    scaled_cell = atoms.cell * factor
    atoms.set_cell(scaled_cell, scale_atoms=True)
    if not cell_defined:
        atoms.cell = None


def get_distance_matrix(atoms):
    n_atoms = len(atoms)
    dists = np.zeros([n_atoms, n_atoms])
    for i, atom_a in enumerate(atoms):
        for j, atom_b in enumerate(atoms):
            dists[i, j] = np.linalg.norm(atom_a.position - atom_b.position)
    return dists


def get_cc_bond_len(atoms):
    c_atoms = [a for a in atoms if a.symbol[0] == "C"]
    n_atoms = len(c_atoms)
    dists = get_distance_matrix(c_atoms)
    # Find bond distances to closest neighbor.
    dists += np.diag([np.inf] * n_atoms)  # Don't consider diagonal.
    bonds = np.amin(dists, axis=1)
    # Average bond distance.
    avg_bond = np.mean(bonds)
    return avg_bond


def scale_to_cc_bond_length(atoms, cc_bond=1.42):
    factor = cc_bond / get_cc_bond_len(atoms)
    scale_atoms(atoms, factor)


def find_neighbors(geom, neighbor_list, depth=1):
    """
    geom - ase atoms object
    depth - up to which order of nearest neighbours to include (e.g. 2 - second nearest)
    
    returns:
    
    neighbors[i_atom] = [[first_nearest_neighbors], [second_nearest_neighbors], ...]
    """

    if depth > 3:
        raise Exception("Only up to 3rd nearest neighbor hoppings are supported.")

    i_arr, j_arr = neighbor_list

    neighbors = [[[]] for i in range(len(geom))]
    already_added = [{i} for i in range(len(geom))]

    # first nearest neighbors
    for i_at, j_at in zip(i_arr, j_arr):
        neighbors[i_at][0].append(j_at)
        already_added[i_at].add(j_at)

    if depth >= 2:
        # second nearest neighbors are just the neighbors of first neighbors
        for i_at in range(len(neighbors)):

            # add new lists for new layer
            neighbors[i_at].append([])

            # go through the "previous layer" and add the new layer
            for cur_n in neighbors[i_at][-2]:
                new_neighbors = set(neighbors[cur_n][0])
                # add new neighbor only if it's not already added for current atom
                add_neighbors = new_neighbors.difference(already_added[i_at])
                already_added[i_at].update(add_neighbors)
                neighbors[i_at][-1] += add_neighbors

    if depth >= 3:
        # topological 3rd nearest neighbors can either be 3rd or 4th NNs in the TB
        # determine this based on the distance in a pristine graphene lattice,
        # where the 3rd NN distance is 2*cc and 4th NN distance is 2.65*cc

        dist_matrix = get_distance_matrix(geom)
        cc_bond = get_cc_bond_len(geom)
        dist_threshold = 2.32 * cc_bond

        for i_at in range(len(neighbors)):

            # add new lists for the 3rd and 4th layer
            neighbors[i_at].append([])
            neighbors[i_at].append([])

            # go through the 2nd layer and fill the new layers
            for cur_n in neighbors[i_at][-3]:
                new_neighbors = neighbors[cur_n][0]

                for nn in new_neighbors:
                    if nn not in already_added[i_at]:
                        dist = dist_matrix[i_at, nn]
                        if dist > dist_threshold:
                            neighbors[i_at][-1].append(nn)
                        else:
                            neighbors[i_at][-2].append(nn)
                        already_added[i_at].add(nn)

    return neighbors


### ------------------------------------------------------------------------------
### PLOTTING TOOLS
### ------------------------------------------------------------------------------


def visualize_backbone(ax, atoms, neighbor_list):
    i_arr, j_arr = neighbor_list
    for i, j in zip(i_arr, j_arr):
        if i < j:
            p1 = atoms.positions[i]
            p2 = atoms.positions[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3.0, solid_capstyle='round')


def visualize_evec(ax, atoms, evec):
    #area = 0.0
    #vol = 0.0
    for at, e in zip(atoms, evec):
        p = at.position

        mod = 1.6 * np.abs(e)  # normalized area
        #mod = np.abs(e)**(2/3) # normalized vol

        phase = np.abs(np.angle(e) / np.pi)
        col = (1.0 - phase, 0.0, phase)
        circ = plt.Circle(p[:2], radius=mod, color=col, zorder=10)
        ax.add_artist(circ)

        #area += np.pi*mod**2
        #vol += 4/3*np.pi*mod**3

    #print("Norm: %.6f" % np.abs(np.sum(evec**2)))
    #print("Area: %.6f"%area)
    #print(" Vol: %.6f"%vol)


def make_evec_plot(ax, atoms, neighbor_list, data, title=None, filename=None):

    ax.set_aspect('equal')
    visualize_backbone(ax, atoms, neighbor_list)
    visualize_evec(ax, atoms, data)
    ax.axis('off')
    xmin = np.min(atoms.positions[:, 0]) - 2.0
    xmax = np.max(atoms.positions[:, 0]) + 2.0
    ymin = np.min(atoms.positions[:, 1]) - 2.0
    ymax = np.max(atoms.positions[:, 1]) + 2.0
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title)

    if filename is not None:
        plt.savefig('%s.png' % filename, dpi=300, bbox_inches='tight')
        plt.savefig('%s.pdf' % filename, bbox_inches='tight')


### ------------------------------------------------------------------------------
### GRID ORBITALS
### ------------------------------------------------------------------------------


def get_local_grid(x_arr, y_arr, p, cutoff=10.0):
    """Method that selects a local grid around an atom

    Args:
        x_arr: global x array
        y_arr: global y array
        p: atomic position
        cutoff (float, optional): extent of local grid in all directions. Defaults to 5.0.
    """

    x_min_i = np.abs(x_arr - p[0] + cutoff).argmin()
    x_max_i = np.abs(x_arr - p[0] - cutoff).argmin()
    y_min_i = np.abs(y_arr - p[1] + cutoff).argmin()
    y_max_i = np.abs(y_arr - p[1] - cutoff).argmin()

    local_x, local_y = np.meshgrid(x_arr[x_min_i:x_max_i], y_arr[y_min_i:y_max_i], indexing='ij')

    return [x_min_i, x_max_i, y_min_i, y_max_i], [local_x, local_y]


def hydrogen_like_orbital(x, y, z, n, l, m, nuc=1):
    # https://en.wikipedia.org/wiki/Hydrogen-like_atom#Non-relativistic_wavefunction_and_energy

    r_grid = np.sqrt(x**2 + y**2 + z**2)
    theta_grid = np.arccos(z / (r_grid + 1e-20))
    phi_grid = np.arctan(y / (x + 1e-20))

    a0 = 0.529177  # Bohr radius in angstrom

    def radial(r, n, l):
        norm_factor = np.sqrt(
            (2 * nuc / (n * a0))**3 * np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l)))
        arg = 2 * nuc * r / (n * a0)
        rad = np.exp(-0.5 * arg) * arg**l * scipy.special.genlaguerre(n - l - 1, 2 * l + 1)(arg)
        return norm_factor * rad

    #pylint: disable=no-member
    orbital = radial(r_grid, n, l) * scipy.special.sph_harm(m, l, phi_grid, theta_grid)
    return orbital


def carbon_2pz_slater(x, y, z, z_eff=3.25):
    """Carbon 2pz slater orbital

    z_eff determines the effective nuclear charge interacting with the pz orbital
    Potential options:

    z_eff = 1
        This corresponds to a hydrogen-like 2pz orbital and in
        some cases matches well with DFT reference

    z_eff = 3.136
        Value shown in https://en.wikipedia.org/wiki/Effective_nuclear_charge

    z_eff = 3.25
        This is the value calculated by Slater's rules (https://en.wikipedia.org/wiki/Slater%27s_rules)
        This value is also used in https://doi.org/10.1038/s41557-019-0316-8
        This is the default.
    
    """
    r_grid = np.sqrt(x**2 + y**2 + z**2)  # angstrom
    a0 = 0.529177  # Bohr radius in angstrom
    return z * np.exp(-z_eff * r_grid / (2 * a0))


def gaussian(x, fwhm):
    sigma = fwhm / 2.3548
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


### ------------------------------------------------------------------------------
### MISC
### ------------------------------------------------------------------------------


def orb_label(i_mo, n_el):
    i_rel = i_mo - n_el + 1
    if i_rel < 0:
        label = "HOMO%d" % i_rel
    elif i_rel == 0:
        label = "HOMO"
    elif i_rel == 1:
        label = "LUMO"
    else:
        label = "LUMO+%d" % (i_rel - 1)
    return label
