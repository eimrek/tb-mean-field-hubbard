import numpy as np

import matplotlib.pyplot as plt

import ase
import ase.neighborlist

import pythtb

from . import utils
from . import mfh_pp

### ------------------------------------------------------------------------------
### Main class
### ------------------------------------------------------------------------------


class MeanFieldHubbardModel:
    """
    Class that takes care of running the Mean Field Hubbard calculation
    """
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

        # OUTPUT VARIABLES:

        self.spin_resolved_dens = None
        self.absmag_iter = None
        self.energy_iter = None

        self.evals = None
        self.evecs = None

        self.const_mfh_energy_term = None
        self.energy = None
        self.abs_mag = None

        self.spin_density = None
        self.gap_a, self.gap_b, self.gap_eff = None, None, None

        # POST-PROCESSING INSTANCE
        self.pp = mfh_pp.MFHPostProcess(self)

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

    def override_hoppings(self, custom_t_list=[]):
        """ Override tight binding hoppings:

        custom_t_list - list of (atom index 1, atom index 2, new_hop)

        Note: adding the hopping in one direction automatically adds the other as well
        """
        for custom_t in custom_t_list:
            if len(custom_t) != 3:
                print("Error: please specify (atom index 1, atom index 2, hopping)")
                return
            i1, i2, t = custom_t
            if i1 == i2:
                print("Error: can't specify hopping from a site to itself.")
                return
            if not (0 <= i1 < self.num_atoms) or not (0 <= i2 < self.num_atoms):
                print("Error: index out of range.")
                return
            if i1 < i2: i1, i2 = i2, i1
            self.model_a.set_hop(-t, i1, i2, mode='reset')
            self.model_b.set_hop(-t, i1, i2, mode='reset')

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