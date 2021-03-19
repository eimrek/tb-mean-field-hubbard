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


def hydrogen_like_orbital(x, y, z, n, l, m, nuc=1):
    # https://en.wikipedia.org/wiki/Hydrogen-like_atom#Non-relativistic_wavefunction_and_energy

    r = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    theta = lambda x, y, z: np.arccos(z / (r(x, y, z) + 1e-20))
    phi = lambda x, y, z: np.arctan(y / (x + 1e-20))

    a0 = 0.529177  # Bohr radius in angstrom

    def radial(r, n, l):
        norm_factor = np.sqrt(
            (2 * nuc / (n * a0))**3 * np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l)))
        rad = np.exp(-nuc * r /
                     (n * a0)) * (2 * nuc * r /
                                  (n * a0))**l * scipy.special.genlaguerre(n - l - 1, 2 * l + 1)(2 * nuc * r / (n * a0))
        return norm_factor * rad

    #pylint: disable=no-member
    orbital = radial(r(x, y, z), n, l) * scipy.special.sph_harm(m, l, phi(x, y, z), theta(x, y, z))
    return orbital


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
