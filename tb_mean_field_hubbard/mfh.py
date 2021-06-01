import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import ase
import ase.neighborlist

import pythtb

from . import utils

### ------------------------------------------------------------------------------
### Main class
### ------------------------------------------------------------------------------


class MeanFieldHubbardModel:
    def __init__(self, ase_geom, t_list=[2.7], charge=0, multiplicity='auto', bond_cutoff='auto'):

        self.t_list = t_list
        self.charge = charge

        if multiplicity == 'auto':
            self.multiplicity = 0
            self.relax_multiplicity = True
        else:
            self.multiplicity = multiplicity
            self.relax_multiplicity = False

        self.ase_geom = ase_geom
        self.num_atoms = len(ase_geom)

        self.figure_size = (np.ptp(self.ase_geom.positions, axis=0)[:2] + 1.0) / 4.0
        self.figure_size[0] = max([self.figure_size[0], 3.5])  # to have enough space for titles

        self.spin_guess = self._load_spin_guess(self.ase_geom)

        if bond_cutoff == 'auto':
            cc_len = utils.get_cc_bond_len(self.ase_geom)
            # graphene 2nd nn distance is 1.73*cc, so use halfway there as the cutoff
            bond_cutoff = 1.37 * cc_len

        self.neighbor_list = ase.neighborlist.neighbor_list('ij', self.ase_geom, bond_cutoff)

        self.neighbors = utils.find_neighbors(self.ase_geom, self.neighbor_list, depth=len(self.t_list))

        self.num_spin_el = None
        self._set_up_tb_model()

        self.absmag_iter = None
        self.energy_iter = None

        self.evals = None
        self.evecs = None

        # natural orbital evals and evecs
        self.no_evals = None
        self.no_evecs = None

    ### ------------------------------------------------------------------------------
    ### TB routines
    ### ------------------------------------------------------------------------------

    def _load_spin_guess(self, ase_geom, flip_alpha_majority=True):
        """
        flip_alpha_majority - flip spin guess to have majority spin in alpha channel
        """
        spin_guess = []
        for at in ase_geom:
            if at.tag == 0:
                spin_guess.append([0.5, 0.5])
            elif at.tag == 1:
                spin_guess.append([1.0, 0.0])
            elif at.tag == 2:
                spin_guess.append([0.0, 1.0])
        spin_guess = np.array(spin_guess)
        if flip_alpha_majority:
            if np.sum(spin_guess[:, 0]) < np.sum(spin_guess[:, 1]):
                spin_guess[:, 0], spin_guess[:, 1] = spin_guess[:, 1], spin_guess[:, 0].copy()
        return spin_guess

    def _set_up_tb_model(self):

        self.num_el = self.num_atoms - self.charge

        if not self.relax_multiplicity:

            if self.multiplicity % 2 == self.num_el % 2:
                raise Exception("ERROR: Charge & multiplicity combination not allowed!")

            # determine spin populations
            self.num_spin_el = [
                (self.num_el + (self.multiplicity - 1)) // 2,
                (self.num_el - (self.multiplicity - 1)) // 2,
            ]

        else:
            self.num_spin_el = [0, 0]

        lat = [[1.0, 0.0], [0.0, 1.0]]

        orb = []
        for at in self.ase_geom:
            orb.append(at.position[:2])

        self.model_a = pythtb.tb_model(0, 2, lat, orb, nspin=1)
        self.model_b = pythtb.tb_model(0, 2, lat, orb, nspin=1)

        for i_at in range(len(self.neighbors)):

            for d in range(len(self.t_list)):

                t = self.t_list[d]

                if t == 0.0:
                    continue

                ns = self.neighbors[i_at][d]

                for n in ns:

                    if n < i_at:
                        self.model_a.set_hop(-t, i_at, n)
                        self.model_b.set_hop(-t, i_at, n)

    def visualize_tb_model(self):
        self.model_a.visualize(0, 1)
        plt.show()

    def visualize_spin_guess(self):
        plt.figure(figsize=self.figure_size)
        spin_guess_plot = 0.5 * (self.spin_guess[:, 0] - self.spin_guess[:, 1])
        utils.make_evec_plot(plt.gca(), self.ase_geom, self.neighbor_list, spin_guess_plot)
        plt.show()

    def print_parameters(self):
        print("Total number of electrons: %d" % self.num_el)
        if not self.relax_multiplicity:
            print("α electrons: %d" % self.num_spin_el[0])
            print("β electrons: %d" % self.num_spin_el[1])

    ### -------------------------------------------------------------------
    ### MFH routines
    ### -------------------------------------------------------------------

    def new_spin_res_dens(self, evals_list, evecs_list):

        mix_coef = 0.8

        new_spin_resolved_dens = (1.0 - mix_coef) * np.copy(self.spin_resolved_dens)

        if not self.relax_multiplicity:

            for i_spin, n_el in enumerate(self.num_spin_el):
                for i_mo in range(n_el):
                    new_spin_resolved_dens[:, i_spin] += mix_coef * np.abs(evecs_list[i_spin][i_mo])**2

        else:
            # in case of relaxed multiplicity, fill the lowest orbitals, without considering spin
            i_a = 0
            i_b = 0
            for _i_el in range(self.num_el):
                en_a = evals_list[0][i_a]
                en_b = evals_list[1][i_b]
                if en_a < en_b:
                    new_spin_resolved_dens[:, 0] += mix_coef * np.abs(evecs_list[0][i_a])**2
                    i_a += 1
                else:
                    new_spin_resolved_dens[:, 1] += mix_coef * np.abs(evecs_list[1][i_b])**2
                    i_b += 1
            self.num_spin_el = [i_a, i_b]

        return new_spin_resolved_dens

    def mfh_iteration(self, u):

        # Update MFH Hubbard on-site potential

        for i_at, d in enumerate(self.spin_resolved_dens):

            self.model_a.set_onsite(u * d[1], i_at, mode="reset")
            self.model_b.set_onsite(u * d[0], i_at, mode="reset")

        # Solve the new TB
        (evals_a, evecs_a) = self.model_a.solve_all(eig_vectors=True)
        (evals_b, evecs_b) = self.model_b.solve_all(eig_vectors=True)

        # Update spin-resolved density
        self.spin_resolved_dens[:] = self.new_spin_res_dens([evals_a, evals_b], [evecs_a, evecs_b])

        evals = np.stack([evals_a, evals_b])
        evecs = np.stack([evecs_a, evecs_b])

        return evals, evecs

    def abs_magnetization(self, spin_resolved_dens):
        return np.sum(np.abs(spin_resolved_dens[:, 0] - spin_resolved_dens[:, 1]))

    def get_total_occ_orb_energy(self, evals):
        energy = 0.0
        for i_spin in range(2):
            for i_ev, ev in enumerate(evals[i_spin]):
                if i_ev < self.num_spin_el[i_spin]:
                    energy += ev
        return energy

    def run_mfh(self, u, print_iter=False, plot=False, energy_tol=1e-6, mag_tol=1e-4, max_iter=100):

        self.spin_resolved_dens = np.copy(self.spin_guess)

        self.absmag_iter = []
        self.energy_iter = []

        for iteration in range(max_iter):
            evals, evecs = self.mfh_iteration(u)
            self.absmag_iter.append(self.abs_magnetization(self.spin_resolved_dens))
            self.energy_iter.append(self.get_total_occ_orb_energy(evals))

            if print_iter:
                print("Iter %3d: energy %.10f; abs mag %.10f" % (iteration, self.energy_iter[-1], self.absmag_iter[-1]))

            if iteration > 0:
                de = self.energy_iter[-1] - self.energy_iter[-2]
                dm = self.absmag_iter[-1] - self.absmag_iter[-2]

                if np.abs(de) < energy_tol and np.abs(dm) < mag_tol:
                    if print_iter:
                        print("Converged after %d iterations!" % iteration)
                    break

                if iteration == max_iter - 1:
                    print("No convergence after %d iterations. dE=%e" % (iteration + 1, de))

        self.evals = evals
        self.evecs = evecs

        if self.relax_multiplicity:
            self.multiplicity = max(self.num_spin_el) - min(self.num_spin_el) + 1

        # post-process

        # constant energy term in the MFH  [ U*sum(<n_i><n_j>) ]
        self.const_mfh_energy_term = -u * np.sum(self.spin_resolved_dens[:, 0] * self.spin_resolved_dens[:, 1])

        self.energy = self.energy_iter[-1] + self.const_mfh_energy_term
        self.abs_mag = self.absmag_iter[-1]

        self.spin_density = self.spin_resolved_dens[:, 0] - self.spin_resolved_dens[:, 1]

        self.gap_a, self.gap_b, self.gap_eff = self.gaps(self.evals)

        if plot:
            ax1 = plt.gca()
            ax1.set_xlabel('iteration')
            ax1.plot(self.absmag_iter, color='blue')
            ax1.set_ylabel('abs. magnetization', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2 = ax1.twinx()
            ax2.plot(self.energy_iter, color='red')
            ax2.set_ylabel('energy', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            plt.show()

    ### -------------------------------------------------------------------
    ### Postprocess
    ### -------------------------------------------------------------------

    def gaps(self, evals):
        try:
            homo_a = evals[0][self.num_spin_el[0] - 1]
            lumo_a = evals[0][self.num_spin_el[0]]

            homo_b = evals[1][self.num_spin_el[1] - 1]
            lumo_b = evals[1][self.num_spin_el[1]]

            gap_a = lumo_a - homo_a
            gap_b = lumo_b - homo_b
            gap = np.min([lumo_a, lumo_b]) - np.max([homo_a, homo_b])

            return gap_a, gap_b, gap

        except IndexError:
            return np.nan, np.nan, np.nan

    def plot_evals(self, evals, filename=None):
        plt.figure(figsize=(3.0, 6))
        ax = plt.gca()

        for i_spin in range(2):
            for i_ev, ev in enumerate(evals[i_spin]):
                col = 'blue'
                if i_ev < self.num_spin_el[i_spin]:
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
        return xmin, xmax, ymin, ymax

    def calc_orb_map(self, evec, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):

        extent = self._get_atoms_extent(self.ase_geom, edge_space)

        # define grid
        x_arr = np.arange(extent[0], extent[1], dx)
        y_arr = np.arange(extent[2], extent[3], dx)

        orb_map = np.zeros((len(x_arr), len(y_arr)), dtype=np.complex)

        for at, coef in zip(self.ase_geom, evec):
            p = at.position
            local_i, local_grid = utils.get_local_grid(x_arr, y_arr, p, cutoff=1.2 * h + 4.0)
            pz_orb = utils.carbon_2pz_slater(local_grid[0] - p[0], local_grid[1] - p[1], h, z_eff)
            orb_map[local_i[0]:local_i[1], local_i[2]:local_i[3]] += coef * pz_orb

        return orb_map

    def plot_orb_squared_map(self, ax, evec, h=10.0, edge_space=5.0, dx=0.1, title=None, cmap='seismic', z_eff=3.25):
        orb_map = np.abs(self.calc_orb_map(evec, h, edge_space, dx, z_eff))**2
        extent = self._get_atoms_extent(self.ase_geom, edge_space)
        ax.imshow(orb_map.T, origin='lower', cmap=cmap, extent=extent)
        ax.axis('off')
        ax.set_title(title)

    def calc_sts_map(self, energy, broadening=0.05, h=10.0, edge_space=5.0, dx=0.1, z_eff=3.25):

        extent = self._get_atoms_extent(self.ase_geom, edge_space)
        # define grid
        x_arr = np.arange(extent[0], extent[1], dx)
        y_arr = np.arange(extent[2], extent[3], dx)

        final_map = np.zeros((len(x_arr), len(y_arr)))

        for i_spin in range(2):
            for i_orb, evl in enumerate(self.evals[i_spin]):
                if np.abs(energy - evl) <= 3.0 * broadening:
                    broad_coef = utils.gaussian(energy - evl, broadening)
                    evec = self.evecs[i_spin][i_orb]
                    orb_ldos_map = np.abs(self.calc_orb_map(evec, h, edge_space, dx, z_eff))**2
                    final_map += broad_coef * orb_ldos_map
        return final_map

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

        final_map = self.calc_sts_map(energy, broadening, h, edge_space, dx, z_eff)

        extent = self._get_atoms_extent(self.ase_geom, edge_space)

        ax.imshow(final_map.T, origin='lower', cmap=cmap, extent=extent)
        ax.axis('off')
        ax.set_title(title)

    def plot_eigenvector(self, ax, evec, title=None):
        utils.make_evec_plot(ax, self.ase_geom, self.neighbor_list, evec, title=title)

    def plot_mo_eigenvector(self, mo_index, spin=0, ax=None):
        title = "mo%d s%d %s, en: %.2f" % (mo_index, spin, utils.orb_label(
            mo_index, self.num_spin_el[spin]), self.evals[spin][mo_index])
        if ax is None:
            plt.figure(figsize=self.figure_size)
            self.plot_eigenvector(plt.gca(), self.evecs[spin][mo_index], title=title)
            plt.show()
        else:
            self.plot_eigenvector(ax, self.evecs[spin][mo_index], title=title)

    def plot_no_eigenvector(self, no_index, ax=None):
        title = "no%d, occ=%.4f" % (no_index, self.no_evals[no_index])
        if ax is None:
            plt.figure(figsize=self.figure_size)
            self.plot_eigenvector(plt.gca(), self.no_evecs[no_index], title=title)
            plt.show()
        else:
            self.plot_eigenvector(ax, self.no_evecs[no_index], title=title)

    def report(self, num_orb=2, sts_h=10.0, sts_broad=0.05):

        print(f"multiplicity:       {self.multiplicity:12d}")
        print(f"abs. magnetization: {self.abs_mag:12.4f}")
        print(f"energy:             {self.energy: 12.4f}")
        print("---")
        print("spin density:")
        plt.figure(figsize=self.figure_size)
        utils.make_evec_plot(plt.gca(), self.ase_geom, self.neighbor_list, self.spin_density)
        plt.show()

        print("---")
        print("eigenvalues:")
        self.plot_evals(self.evals)

        gap_a, gap_b, gap = self.gaps(self.evals)
        print(f"gap alpha: {gap_a:.4f}")
        print(f"gap beta:  {gap_b:.4f}")
        print(f"gap eff.:  {gap:.4f}")

        print("---")
        print("frontier orbitals:")

        for i_rel in np.arange(num_orb, -num_orb, -1):

            i_mo = int(np.around(0.5 * (self.num_spin_el[0] + self.num_spin_el[1]))) + i_rel - 1

            if i_mo < 0 or i_mo > len(self.evecs[0]) - 1:
                continue

            _fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(4 * self.figure_size[0], self.figure_size[1]))

            self.plot_mo_eigenvector(i_mo, spin=0, ax=axs[0])
            self.plot_mo_eigenvector(i_mo, spin=1, ax=axs[1])

            title1 = "sts h=%.1f, en: %.2f" % (sts_h, self.evals[0][i_mo])
            self.plot_sts_map(axs[2], self.evals[0][i_mo], broadening=sts_broad, h=sts_h, title=title1)

            title2 = "sts h=%.1f, en: %.2f" % (sts_h, self.evals[1][i_mo])
            self.plot_sts_map(axs[3], self.evals[1][i_mo], broadening=sts_broad, h=sts_h, title=title2)

            plt.show()

    def calculate_natural_orbitals(self):
        # build the one particle reduced density matrix
        dens_mat = None

        for i_spin in range(2):
            for i_el in range(self.num_spin_el[i_spin]):
                evec = self.evecs[i_spin, i_el]
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
