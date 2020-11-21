
from pythtb import *
import numpy as np

import matplotlib.pyplot as plt

import os

import ase
import ase.io
import ase.neighborlist
import ase.visualize.plot

### ------------------------------------------------------------------------------
### GEOMETRY TOOLS
### ------------------------------------------------------------------------------

def find_neighbors(geom, cutoff=1.8, depth=1):
    """
    geom - ase atoms object
    depth - up to which order of nearest neighbours to include (e.g. 2 - second nearest)
    
    returns:
    
    neighbors[i_atom] = [[first_nearest_neighbors], [second_nearest_neighbors], ...]
    """
    
    i_arr, j_arr = ase.neighborlist.neighbor_list('ij', geom, cutoff)
    
    neighbors = [[[]] for i in range(len(geom))]
    already_added = [{i} for i in range(len(geom))]
    
    # first nearest neighbors
    for i_at, j_at in zip(i_arr, j_arr):
        neighbors[i_at][0].append(j_at)
        already_added[i_at].add(j_at)
    
    # n-th nearest neighbors
    for d in range(depth-1):
        
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

def visualize_backbone(ax, atoms):
    i_arr, j_arr = ase.neighborlist.neighbor_list('ij', atoms, 1.8)
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
        
        mod = 1.6*np.abs(e) # normalized area
        #mod = np.abs(e)**(2/3) # normalized vol
        
        phase = np.angle(e)/np.pi
        col = (1.0-phase, 0.0, phase)
        circ = plt.Circle(p[:2], radius=mod, color=col, zorder=10)
        ax.add_artist(circ)
        
        #area += np.pi*mod**2
        #vol += 4/3*np.pi*mod**3
        
    #print("Norm: %.6f" % np.abs(np.sum(evec**2)))
    #print("Area: %.6f"%area)
    #print(" Vol: %.6f"%vol)

def make_plot(figsize, atoms, data_list, title_list=None, filename=None):

    if not isinstance(data_list, list):
        raise Exception("data_list needs to be a list")

    figs = (figsize[0] * len(data_list), figsize[1])
    
    fig, _ = plt.subplots(nrows=1, ncols=len(data_list), figsize=figs)
    axs = fig.axes

    for i_data, d in enumerate(data_list):
        ax = axs[i_data]

        title = None
        if title_list is not None:
            title = title_list[i_data]

        ax.set_aspect('equal')

        visualize_backbone(ax, atoms)
        visualize_evec(ax, atoms, d)
        ax.axis('off')
        xmin = np.min(atoms.positions[:, 0])-2.0
        xmax = np.max(atoms.positions[:, 0])+2.0
        ymin = np.min(atoms.positions[:, 1])-2.0
        ymax = np.max(atoms.positions[:, 1])+2.0
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(title)
    if filename is not None:
        plt.savefig('%s.png' % filename, dpi=300, bbox_inches='tight')
        plt.savefig('%s.pdf' % filename, bbox_inches='tight')
    plt.show()
    
def orb_label(i_mo, n_el):
    i_rel = i_mo - n_el + 1
    if i_rel < 0:
        label = "HOMO%d"%i_rel
    elif i_rel == 0:
        label = "HOMO"
    elif i_rel == 1:
        label = "LUMO"
    else:
        label = "LUMO+%d"%(i_rel-1)
    return label


class MeanFieldHubbardModel:

    def __init__(self, ase_geom, t_list = [-2.7], charge = 0, multiplicity = 1):
        
        self.t_list = t_list
        self.multiplicity = multiplicity
        self.charge = charge

        self.ase_geom = ase_geom
        self.num_atoms = len(ase_geom)

        self.figure_size = self._atoms_extent(self.ase_geom) / 4.0
        self.figure_size[0] += 2.5

        self.spin_guess = self._load_spin_guess(self.ase_geom)
        
        self.neighbors = find_neighbors(self.ase_geom, 1.8, len(self.t_list))

        self._set_up_tb_model()

        self.absmag_iter = None
        self.energy_iter = None

        self.evals = None
        self.evecs = None

    def _atoms_extent(self, ase_geom):
        x_extent = np.ptp(ase_geom.positions[:, 0]) + 1.0
        y_extent = np.ptp(ase_geom.positions[:, 1]) + 1.0
        return np.array([x_extent, y_extent])

    def _load_spin_guess(self, ase_geom):
        spin_guess = []
        for at in ase_geom:
            if at.tag == 0:
                spin_guess.append([0.5, 0.5])
            elif at.tag == 1:
                spin_guess.append([1.0, 0.0])
            elif at.tag == 2:
                spin_guess.append([0.0, 1.0])
        return np.array(spin_guess)
        
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

        lat = [
            [1.0,0.0],
            [0.0,1.0]
        ]

        orb = []
        for at in self.ase_geom:
            orb.append(at.position[:2])

        self.model_a = tb_model(0,2,lat,orb, nspin=1)
        self.model_b = tb_model(0,2,lat,orb, nspin=1)

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
        make_plot(self.figure_size, self.ase_geom, [0.5*(self.spin_guess[:, 0] - self.spin_guess[:, 1])])
        
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
                
                new_spin_resolved_dens[:, i_spin] += mix_coef * np.abs(evecs_list[i_spin][i_mo]) ** 2
        
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
                print("Iter %3d: energy %.10f; abs mag %.10f" % (
                    iteration, self.energy_iter[-1], self.absmag_iter[-1]))

            if iteration > 0:
                de = self.energy_iter[-1] - self.energy_iter[-2]
                dm = self.absmag_iter[-1] - self.absmag_iter[-2]

                if np.abs(de) < energy_tol and np.abs(dm) < mag_tol:
                    if print_iter:
                        print("Converged after %d iterations!" % iteration)
                    break

                if iteration == max_iter-1:
                    print("No convergence after %d iterations. dE=%e" % (iteration+1, de))

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
            homo_a = evals[0][self.num_spin_el[0]-1]
            lumo_a = evals[0][self.num_spin_el[0]]
            
            homo_b = evals[1][self.num_spin_el[1]-1]
            lumo_b = evals[1][self.num_spin_el[1]]
            
            gap_a = lumo_a-homo_a
            gap_b = lumo_b-homo_b
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

        ax.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.xlim(0.0, 2.0)
        #plt.ylim(-6.0, 10.0)
        plt.ylabel("Energy (eV)")
        #plt.yticks(fontsize=30)
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def report(self, num_orb=2):

        print("abs. magnetization: %14.6f" % self.abs_mag)
        print("energy:             %14.6f" % self.energy)
        print("---")
        print("spin density:")
        make_plot(self.figure_size, self.ase_geom, [1.0*self.spin_density])

        print("---")
        print("eigenvalues:")
        self.plot_evals(self.evals)

        gap_a, gap_b, gap = self.gaps(self.evals)
        print("gap alpha: %.6f" % gap_a)
        print("gap beta:  %.6f" % gap_b)
        print("gap eff.:  %.6f" % gap)

        for i_rel in np.arange(num_orb, -num_orb, -1):
            
            i_mo = int(np.around(0.5*(self.num_spin_el[0] + self.num_spin_el[1]))) + i_rel - 1

            if i_mo < 0 or i_mo > len(self.evecs[0])-1:
                continue
            
            titles = [
                "mo%d α %s, en: %.2f" % (i_mo, orb_label(i_mo, self.num_spin_el[0]), self.evals[0][i_mo]),
                "mo%d β %s, en: %.2f" % (i_mo, orb_label(i_mo, self.num_spin_el[1]), self.evals[1][i_mo]),
            ]
            make_plot(self.figure_size, self.ase_geom, [self.evecs[0][i_mo], self.evecs[1][i_mo]], title_list=titles)

    def plot_orbital(self, mo_index, spin=0):
        title = "mo%d s%d %s, en: %.2f" % (
            mo_index, spin, orb_label(mo_index, self.num_spin_el[spin]), self.evals[spin][mo_index])
        make_plot(self.figure_size, self.ase_geom, [self.evecs[spin][mo_index]], title_list=[title])

