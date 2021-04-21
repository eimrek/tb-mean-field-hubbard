import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os

import ase
import ase.neighborlist

import pythtb

import sys

from . import utils

### ------------------------------------------------------------------------------
### Main class
### ------------------------------------------------------------------------------


class MeanFieldHubbardModel:
    def __init__(self, ase_geom, t_list=[-2.7], charge=0, multiplicity=1, bond_cutoff=1.8):

        self.t_list = t_list
        self.multiplicity = multiplicity
        self.charge = charge

        self.ase_geom = ase_geom
        self.num_atoms = len(ase_geom)

        self.figure_size = utils.atoms_extent(self.ase_geom) / 4.0
        self.figure_size[0] += 2.5

        self.spin_guess = self._load_spin_guess(self.ase_geom)

        self.neighbor_list = ase.neighborlist.neighbor_list('ij', self.ase_geom, bond_cutoff)

        self.neighbors = utils.find_neighbors(self.ase_geom, self.neighbor_list, depth=len(self.t_list))

        self.num_spin_el = None
        self._set_up_tb_model()

        self.absmag_iter = None
        self.energy_iter = None

        self.evals = None
        self.evecs = None

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

        # check parameter validity:

        self.num_el = self.num_atoms - self.charge

        if self.multiplicity % 2 == self.num_el % 2:
            raise Exception("ERROR: Charge & multiplicity combination not allowed!")

        # determine spin populations

        self.num_spin_el = [
            (self.num_el + (self.multiplicity - 1)) // 2,
            (self.num_el - (self.multiplicity - 1)) // 2,
        ]

        lat = [[1.0, 0.0], [0.0, 1.0]]

        orb = []
        for at in self.ase_geom:
            orb.append(at.position[:2])

        self.model_a = pythtb.tb_model(0, 2, lat, orb, nspin=1)
        self.model_b = pythtb.tb_model(0, 2, lat, orb, nspin=1)

        for i_at in range(len(self.neighbors)):

            for d in range(len(self.neighbors[i_at])):

                t = self.t_list[d]

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
        utils.make_plot(plt.gca(), self.ase_geom, self.neighbor_list, spin_guess_plot)
        plt.show()

    def print_parameters(self):
        print("Total number of electrons: %d" % self.num_el)
        print("α electrons: %d" % self.num_spin_el[0])
        print("β electrons: %d" % self.num_spin_el[1])

    ### -------------------------------------------------------------------
    ### MFH routines
    ### -------------------------------------------------------------------

    def new_spin_res_dens(self, evals_list, evecs_list):

        mix_coef = 0.8

        new_spin_resolved_dens = (1.0 - mix_coef) * np.copy(self.spin_resolved_dens)

        for i_spin, n_el in enumerate(self.num_spin_el):

            for i_mo in range(n_el):

                new_spin_resolved_dens[:, i_spin] += mix_coef * np.abs(evecs_list[i_spin][i_mo])**2

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

    def calc_orb_ldos_map(self, i_spin, i_orb, h=6.0, edge_space=5.0, dx=0.1, z_eff=1):

        atoms = self.ase_geom
        xmin = np.min(atoms.positions[:, 0]) - edge_space
        xmax = np.max(atoms.positions[:, 0]) + edge_space
        ymin = np.min(atoms.positions[:, 1]) - edge_space
        ymax = np.max(atoms.positions[:, 1]) + edge_space

        # define grid
        x_arr = np.arange(xmin, xmax, dx)
        y_arr = np.arange(ymin, ymax, dx)

        evc = self.evecs[i_spin][i_orb]
        orb_map = np.zeros((len(x_arr), len(y_arr)), dtype=np.complex)
        for at, coef in zip(self.ase_geom, evc):
            p = at.position
            local_i, local_grid = utils.get_local_grid(x_arr, y_arr, p, cutoff=1.2 * h + 4.0)
            pz_orb = utils.carbon_2pz_slater(local_grid[0] - p[0], local_grid[1] - p[1], h, z_eff)
            orb_map[local_i[0]:local_i[1], local_i[2]:local_i[3]] += coef * pz_orb

        return np.abs(orb_map)**2

    def calc_sts_map(self, energy, broadening=0.05, h=6.0, edge_space=5.0, dx=0.1, z_eff=1):

        final_map = None

        for i_spin in range(2):
            for i_orb, evl in enumerate(self.evals[i_spin]):
                if np.abs(energy - evl) <= 3.0 * broadening:
                    broad_coef = utils.gaussian(energy - evl, broadening)
                    orb_ldos_map = self.calc_orb_ldos_map(i_spin, i_orb, h, edge_space, dx, z_eff)
                    if final_map is None:
                        final_map = broad_coef * orb_ldos_map
                    else:
                        final_map += broad_coef * orb_ldos_map
        return final_map

    def plot_sts_map(self,
                     ax,
                     energy,
                     broadening=0.05,
                     h=6.0,
                     edge_space=5.0,
                     dx=0.1,
                     title=None,
                     cmap='seismic',
                     z_eff=1):

        final_map = self.calc_sts_map(energy, broadening, h, edge_space, dx, z_eff)

        ax.imshow(final_map.T, origin='lower', cmap=cmap)
        ax.axis('off')
        ax.set_title(title)

    def report(self, num_orb=2, sts_h=3.5, sts_broad=0.05):

        print("abs. magnetization: %14.6f" % self.abs_mag)
        print("energy:             %14.6f" % self.energy)
        print("---")
        print("spin density:")
        plt.figure(figsize=self.figure_size)
        utils.make_plot(plt.gca(), self.ase_geom, self.neighbor_list, self.spin_density)
        plt.show()

        print("---")
        print("eigenvalues:")
        self.plot_evals(self.evals)

        gap_a, gap_b, gap = self.gaps(self.evals)
        print("gap alpha: %.6f" % gap_a)
        print("gap beta:  %.6f" % gap_b)
        print("gap eff.:  %.6f" % gap)

        print("---")
        print("frontier orbitals:")

        for i_rel in np.arange(num_orb, -num_orb, -1):

            i_mo = int(np.around(0.5 * (self.num_spin_el[0] + self.num_spin_el[1]))) + i_rel - 1

            if i_mo < 0 or i_mo > len(self.evecs[0]) - 1:
                continue

            _fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(4 * self.figure_size[0], self.figure_size[1]))

            title1 = "mo%d α %s, en: %.2f" % (i_mo, utils.orb_label(i_mo, self.num_spin_el[0]), self.evals[0][i_mo])
            utils.make_plot(axs[0], self.ase_geom, self.neighbor_list, self.evecs[0][i_mo], title1)

            title2 = "mo%d β %s, en: %.2f" % (i_mo, utils.orb_label(i_mo, self.num_spin_el[1]), self.evals[1][i_mo])
            utils.make_plot(axs[1], self.ase_geom, self.neighbor_list, self.evecs[1][i_mo], title2)

            title3 = "sts h=%.1f, en: %.2f" % (sts_h, self.evals[0][i_mo])
            self.plot_sts_map(axs[2], self.evals[0][i_mo], broadening=sts_broad, h=sts_h, title=title3)

            title4 = "sts h=%.1f, en: %.2f" % (sts_h, self.evals[1][i_mo])
            self.plot_sts_map(axs[3], self.evals[1][i_mo], broadening=sts_broad, h=sts_h, title=title4)

            plt.show()

    def plot_orbital(self, mo_index, spin=0):
        title = "mo%d s%d %s, en: %.2f" % (mo_index, spin, utils.orb_label(
            mo_index, self.num_spin_el[spin]), self.evals[spin][mo_index])
        plt.figure(figsize=self.figure_size)
        utils.make_plot(plt.gca(), self.ase_geom, self.neighbor_list, self.evecs[spin][mo_index], title=title)
        plt.show()
