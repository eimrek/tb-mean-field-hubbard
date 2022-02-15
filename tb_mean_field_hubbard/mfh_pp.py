import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from . import utils


class MFHPostProcess:
    """
    Runs various post-processing and visualization on a converged MeanFieldHubbardModel
    """
    def __init__(self, mfh_model):

        self.mfh = mfh_model

        # natural orbital evals and evecs
        self.no_evals = None
        self.no_evecs = None

    def plot_evals(self, evals, filename=None):
        plt.figure(figsize=(3.0, 6))
        ax = plt.gca()

        for i_spin in range(2):
            for i_ev, ev in enumerate(evals[i_spin]):
                col = 'blue'
                if i_ev < self.mfh.num_spin_el[i_spin]:
                    col = 'red'
                line_pos = [0.1, 0.9] if i_spin == 0 else [1.1, 1.9]
                plt.plot(line_pos, [ev, ev], '-', color=col, lw=2.0, solid_capstyle='round')

        blue_patch = mpatches.Patch(color='blue', label='empty')
        red_patch = mpatches.Patch(color='red', label='occupied')
        plt.legend(handles=[blue_patch, red_patch], loc='upper left', bbox_to_anchor=(1.0, 1.0))

        #ax.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        plt.xlim(0.0, 2.0)
        plt.xticks([0.5, 1.5], ["spin α", "spin β"])
        #plt.ylim(-6.0, 10.0)
        plt.ylabel("Energy (eV)")
        #plt.yticks(fontsize=30)
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def _get_atoms_extent(self, atoms, edge_space):
        xmin = np.min(atoms.positions[:, 0]) - edge_space
        xmax = np.max(atoms.positions[:, 0]) + edge_space
        ymin = np.min(atoms.positions[:, 1]) - edge_space
        ymax = np.max(atoms.positions[:, 1]) + edge_space
        return [xmin, xmax, ymin, ymax]

    def calc_orb_map(self, evec, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):

        extent = self._get_atoms_extent(self.mfh.ase_geom, edge_space)

        # define grid
        x_arr = np.arange(extent[0], extent[1], dx)
        y_arr = np.arange(extent[2], extent[3], dx)

        # update extent so that it matches with grid size
        extent[1] = x_arr[-1] + dx
        extent[3] = y_arr[-1] + dx

        orb_map = np.zeros((len(x_arr), len(y_arr)), dtype=np.complex)

        for at, coef in zip(self.mfh.ase_geom, evec):
            p = at.position
            local_i, local_grid = utils.get_local_grid(x_arr, y_arr, p, cutoff=1.2 * h + 4.0)
            pz_orb = utils.carbon_2pz_slater(local_grid[0] - p[0], local_grid[1] - p[1], h, z_eff)
            orb_map[local_i[0]:local_i[1], local_i[2]:local_i[3]] += coef * pz_orb

        return orb_map, extent

    def plot_orb_squared_map(self, ax, evec, h=10.0, edge_space=5.0, dx=0.1, title=None, cmap='seismic', z_eff=3.25):
        orb_map, extent = self.calc_orb_map(evec, h, edge_space, dx, z_eff)
        ax.imshow((np.abs(orb_map)**2).T, origin='lower', cmap=cmap, extent=extent)
        ax.axis('off')
        ax.set_title(title)

    def calc_sts_map(self, energy, broadening=0.05, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):

        final_map = None
        extent = None

        for i_spin in range(2):
            for i_orb, evl in enumerate(self.mfh.evals[i_spin]):
                if np.abs(energy - evl) <= 3.0 * broadening:
                    broad_coef = utils.gaussian(energy - evl, broadening)
                    evec = self.mfh.evecs[i_spin][i_orb]
                    orb_map, extent = self.calc_orb_map(evec, h, edge_space, dx, z_eff)
                    if final_map is None:
                        final_map = broad_coef * np.abs(orb_map)**2
                    else:
                        final_map += broad_coef * np.abs(orb_map)**2

        return final_map, extent

    def plot_sts_map(self,
                     ax,
                     energy,
                     broadening=0.05,
                     h=10.0,
                     edge_space=5.0,
                     dx=0.1,
                     title=None,
                     cmap='seismic',
                     z_eff=3.25):

        final_map, extent = self.calc_sts_map(energy, broadening, h, edge_space, dx, z_eff)

        ax.imshow(final_map.T, origin='lower', cmap=cmap, extent=extent)
        ax.axis('off')
        ax.set_title(title)

    def plot_eigenvector(self, ax, evec, title=None):
        utils.make_evec_plot(ax, self.mfh.ase_geom, self.mfh.neighbor_list, evec, title=title)

    def plot_mo_eigenvector(self, mo_index, spin=0, ax=None):
        title = "mo%d s%d %s, en: %.2f" % (mo_index, spin, utils.orb_label(
            mo_index, self.mfh.num_spin_el[spin]), self.mfh.evals[spin][mo_index])
        if ax is None:
            plt.figure(figsize=self.mfh.figure_size)
            self.plot_eigenvector(plt.gca(), self.mfh.evecs[spin][mo_index], title=title)
            plt.show()
        else:
            self.plot_eigenvector(ax, self.mfh.evecs[spin][mo_index], title=title)

    def plot_no_eigenvector(self, no_index, ax=None):
        title = "no%d, occ=%.4f" % (no_index, self.no_evals[no_index])
        if ax is None:
            plt.figure(figsize=self.mfh.figure_size)
            self.plot_eigenvector(plt.gca(), self.no_evecs[no_index], title=title)
            plt.show()
        else:
            self.plot_eigenvector(ax, self.no_evecs[no_index], title=title)

    def report(self, num_orb=2, sts_h=10.0, sts_broad=0.05):

        print(f"multiplicity:       {self.mfh.multiplicity:12d}")
        print(f"abs. magnetization: {self.mfh.abs_mag:12.4f}")
        print(f"energy:             {self.mfh.energy: 12.4f}")
        print("---")
        print("spin density:")
        plt.figure(figsize=self.mfh.figure_size)
        utils.make_evec_plot(plt.gca(), self.mfh.ase_geom, self.mfh.neighbor_list, self.mfh.spin_density)
        plt.show()

        print("---")
        print("eigenvalues:")
        self.plot_evals(self.mfh.evals)

        print(f"gap alpha: {self.mfh.gap_a:.4f}")
        print(f"gap beta:  {self.mfh.gap_b:.4f}")
        print(f"gap eff.:  {self.mfh.gap_eff:.4f}")

        print("---")
        print("frontier orbitals:")

        for i_rel in np.arange(num_orb, -num_orb, -1):

            i_mo = int(np.around(0.5 * (self.mfh.num_spin_el[0] + self.mfh.num_spin_el[1]))) + i_rel - 1

            if i_mo < 0 or i_mo > len(self.mfh.evecs[0]) - 1:
                continue

            _fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(4 * self.mfh.figure_size[0], self.mfh.figure_size[1]))

            self.plot_mo_eigenvector(i_mo, spin=0, ax=axs[0])
            self.plot_mo_eigenvector(i_mo, spin=1, ax=axs[1])

            title1 = "sts h=%.1f, en: %.2f" % (sts_h, self.mfh.evals[0][i_mo])
            self.plot_sts_map(axs[2], self.mfh.evals[0][i_mo], broadening=sts_broad, h=sts_h, title=title1)

            title2 = "sts h=%.1f, en: %.2f" % (sts_h, self.mfh.evals[1][i_mo])
            self.plot_sts_map(axs[3], self.mfh.evals[1][i_mo], broadening=sts_broad, h=sts_h, title=title2)

            plt.show()

    def calculate_natural_orbitals(self):
        # build the one particle reduced density matrix
        dens_mat = None

        for i_spin in range(2):
            for i_el in range(self.mfh.num_spin_el[i_spin]):
                evec = self.mfh.evecs[i_spin, i_el]
                if dens_mat is None:
                    dens_mat = np.outer(evec, np.conj(evec))
                else:
                    dens_mat += np.outer(evec, np.conj(evec))

        # Diagonalize the density matrix
        self.no_evals, self.no_evecs = np.linalg.eig(dens_mat)
        self.no_evals = np.abs(self.no_evals)
        self.no_evecs = self.no_evecs.T

        # sort the natural orbitals based on occupations
        sort_inds = (-1 * self.no_evals).argsort()
        self.no_evals = self.no_evals[sort_inds]
        self.no_evecs = self.no_evecs[sort_inds]
