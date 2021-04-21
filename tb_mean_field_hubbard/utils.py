import numpy as np
import scipy.special

import matplotlib.pyplot as plt

### ------------------------------------------------------------------------------
### GEOMETRY TOOLS
### ------------------------------------------------------------------------------


def atoms_extent(ase_geom):
    x_extent = np.ptp(ase_geom.positions[:, 0]) + 1.0
    y_extent = np.ptp(ase_geom.positions[:, 1]) + 1.0
    return np.array([x_extent, y_extent])


def find_neighbors(geom, neighbor_list, depth=1):
    """
    geom - ase atoms object
    depth - up to which order of nearest neighbours to include (e.g. 2 - second nearest)
    
    returns:
    
    neighbors[i_atom] = [[first_nearest_neighbors], [second_nearest_neighbors], ...]
    """

    i_arr, j_arr = neighbor_list

    neighbors = [[[]] for i in range(len(geom))]
    already_added = [{i} for i in range(len(geom))]

    # first nearest neighbors
    for i_at, j_at in zip(i_arr, j_arr):
        neighbors[i_at][0].append(j_at)
        already_added[i_at].add(j_at)

    # n-th nearest neighbors
    for _d in range(depth - 1):

        # add new lists for new layer
        for i_at in range(len(neighbors)):
            neighbors[i_at].append([])

        for i_at in range(len(neighbors)):

            # go through the "previous layer" and add the new layer
            for cur_n in neighbors[i_at][-2]:

                new_neighbors = set(neighbors[cur_n][0])

                # add new neighbor only if it's not already added for current atom
                add_neighbors = new_neighbors.difference(already_added[i_at])

                #print(cur_n, add_neighbors)

                already_added[i_at].update(add_neighbors)
                neighbors[i_at][-1] += add_neighbors

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


def make_plot(ax, atoms, neighbor_list, data, title=None, filename=None):

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


def carbon_2pz_slater(x, y, z, z_eff=1):
    """Carbon 2pz slater orbital

    z_eff determines the effective nuclear charge interacting with the pz orbital
    Options:

    z_eff = 1
        This corresponds to a hydrogen-like 2pz orbital
        it matches best with DFT reference and is thus the default

    z_eff = 2.55
        This is the value calculated by Slater's rules
        https://en.wikipedia.org/wiki/Slater%27s_rules

    z_eff = 3.25
        This value matches with https://doi.org/10.1038/s41557-019-0316-8
    
    """
    r_grid = np.sqrt(x**2 + y**2 + z**2)  # angstrom
    a0 = 0.529177  # Bohr radius in angstrom
    return z * np.exp(-z_eff * r_grid * (2 * a0))


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
