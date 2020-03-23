import os
import sys
import yaml
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from numpy.linalg import norm
from collections import OrderedDict
from itertools import combinations
from itertools import product
from matplotlib import rc

import warnings
warnings.filterwarnings("ignore")

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 10
})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')

# the following two functions and setup_yaml() are for preserving the
# order of the dictionary to be printed to the trajectory yaml file


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())


def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


setup_yaml()


class Initialization:
    def __init__(self, param):
        # note that self.param is the parameters specified in the file
        # and is indepdent of whether the parameter value is later changed externally
        with open(param) as ymlfile:
            try:
                self.param = yaml.load(ymlfile, Loader=yaml.FullLoader)
            except AttributeError:
                self.param = yaml.load(ymlfile)

        for attr in self.param:
            setattr(self, attr, self.param[attr])

        # Step 1: Examine the simulation parameters
        # parameters that must be specified i the parameter file
        self.examine_params('simulation', True)
        self.examine_params('N_particles', True)
        self.examine_params('N_steps', True)
        self.examine_params('potential', True)

        if self.simulation != 'MC' and self.simulation != 'MD':
            print('Error: The simulation should be either MC or MD.')
            sys.exit()

        if self.potential == 'central':
            self.examine_params('n', True)
            self.examine_params('u', True)
            self.examine_params('k', True)
            self.examine_params('a', True)

        if self.simulation == 'MC':
            self.examine_params('max_d', True)

        # some special cases
        if self.simulation == 'MC' or (self.simulation == 'MD' and self.velo_method == 'maxwell'):
            # temeprature must be specified in an NVT MC simulation or if 'maxwell' is used
            self.examine_params('t_ref', True)

        if self.simulation == 'MD':
            if 't_ref' not in self.param:
                self.velo_method = 'random'
            self.examine_params('velo_method', False, 'random')

        # Note that if both box_length and rho are specified, box_length will be used in the initialization
        if 'box_length' not in self.param and 'rho' not in self.param:
            print('Error: At least one of the the length of the box or particle density should be specified.')
            sys.exit()

        if 'box_length' in self.param and 'rho' in self.param:
            # make box_length have higher prioirty than rho
            self.rho = self.N_particles / (self.box_length) ** self.dimension

        if 'box_length' not in self.param and 'rho' in self.param:
            self.box_length = (self.N_particles /
                               self.rho) ** (1 / self.dimension)

        if 'box_length' in self.param and 'rho' not in self.param:
            self.rho = self.N_particles / (self.box_length) ** self.dimension

        # assign defaults to non-specified paramters
        if self.simulation == 'MD':
            self.examine_params('dt', False, 0.01)

        if self.potential == 'LJ' or self.potential == 'WCA':
            self.examine_params('epsilon', False, 1)
            self.examine_params('sigma', False, 1)
            self.r_min = 2 ** (1/6) * self.sigma

        self.examine_params('print_freq', False, 1)
        self.examine_params('dimension', False, 3)
        self.examine_params('coords_method', False, 'random')
        self.examine_params('PBC', False, 'yes')
        self.examine_params('m', False, 1)
        self.examine_params('kb', False, 1)
        self.examine_params('energy_truncation', False, 'no')
        self.examine_params('shift_energy', False, 'no')
        self.examine_params('tcoupl', False, 'no')

        if self.tcoupl == 'Andersen':
            self.examine_params('mu', False, 5)
            self.examine_params('nst_toupl', False, 100)
            self.examine_params('t_ref', True)

        if self.energy_truncation == 'yes':
            self.examine_params('r_c', False, 0.5 * self.box_length)

        if self.potential == 'WCA':
            self.tail_correction = 'no'   # no correction for WCA
            self.energy_truncation = 'yes'
            self.shift_energy = 'yes'
            self.r_c = self.r_min
            if self.box_length < 2 * self.r_c:
                print('Error: WCA potential is used but L is smaller than 2 * r_c.')
                print('box_length: %s, r_c: %s' % (self.box_length, self.r_c))
                sys.exit()

        self.examine_params('search_method', False, 'all-pairs')
        if self.search_method == 'verlet':
            self.examine_params('delta', False, 0.2)
            self.r_v = self.delta + self.r_c
            if self.delta + self.r_c >= self.box_length * 0.5:  # r_v should < L/2
                print('Error! Cutoff distance for Verlet list (r_v) should be smaller than L/2.')
                print('r_c = %5.3f, delta = %5.3f, r_v = r_c + delta = %5.3f, L/2 = %5.3f' %(self.r_c, self.delta, self.r_v, 0.5 * self.box_length))
                sys.exit()

        if self.search_method == 'cell':
            self.examine_params('r_cell', False, self.r_c * 1.1)
            self.n_cell = int(np.ceil(self.box_length / self.r_cell)) # number of cells per side

        # print options
        self.examine_params('print_param', False, 'no')
        self.examine_params('print_coords', False, 'yes')
        self.examine_params('print_Ek', False, 'yes')
        self.examine_params('print_Ep', False, 'yes')
        self.examine_params('print_Etotal', False, 'yes')
        self.examine_params('print_temp', False, 'yes')
        self.examine_params('print_pressure', False, 'yes')
        self.examine_params('print_L_total', False, 'yes')

        # Step 2: Print the adopted parameters to a yaml file
        if self.print_param == 'yes':        
            adopted_params = copy.deepcopy(vars(self))
            del adopted_params['param']
            out_param_name = param.split('.')[0] + '_adopted_params.yml'

            for file in os.listdir('.'):
                if file == out_param_name:
                    # make the the output file is newly made
                    os.remove(out_param_name)

            out_params = open(out_param_name, 'a+', newline='')
            out_params.write('# Adopted simulation parameters\n')
            yaml.dump(adopted_params, out_params, default_flow_style=False)

        # Step 3: Add other attributes
        self.prefix = param.split('.')[0]
        self.traj_name = self.prefix + '_traj.yml'

        # Step 4: Delete self.param to prevent confusion/repeptition
        delattr(self, 'param')
    """
    @property
    def box_length(self):
        if 'box_length' not in self.param and 'rho' in self.param:
            return (self.N_particles / self.rho) ** (1 / 3)

    @property
    def rho(self):
        if 'box_length' in self.param and 'rho' not in self.param:
            return self.N_particles / (self.box_length) ** 3
    """

    def examine_params(self, var: str, required=False, default=None):
        if required is True:
            if var not in self.param:
                print('Error: Parameter "%s" must be specified!' % var)
                sys.exit()
        else:
            if var not in self.param:
                setattr(self, var, default)

    def init_coords(self):
        if self.coords_method == 'random':
            self.coords = (0.5 - np.random.rand(self.N_particles,
                                                self.dimension)) * self.box_length  # initial coordinates
        elif self.coords_method == 'lattice':
            # number of grids per side of the lattice
            N = np.ceil((self.N_particles)**(1 / self.dimension)) # number of particles per side
            r_min = -self.box_length / 2
            r_max = self.box_length / 2
            self.d = self.box_length / N   # initial spacing between particles
            pos = np.linspace(r_min + 0.5 * self.d,
                              r_max - 0.5 * self.d, int(N))
            # length is larger or equil to self.coords
            coords_list = list(product(pos, repeat=self.dimension))
            self.coords = np.zeros([self.N_particles, self.dimension])
            for i in range(len(self.coords)):
                self.coords[i] = list(coords_list[i])
        else:
            print(
                'Error: The method for initializing the coordinates should be either "random" or "lattice".')
            sys.exit()

    def init_velo(self):
        if self.velo_method == 'random':
            self.velocities = np.random.rand(
                self.N_particles, self.dimension) * self.box_length * 0.1  # initial velocity

        elif self.velo_method == 'maxwell':
            # reference: http://www.cchem.berkeley.edu/chem195/_l_j___andersen_thermostat_8m.html
            # initialize velocities by drawing samples from Maxwell-Boltzmann distribution
            sigma = np.sqrt(self.t_ref / self.m)   # on the other hand, mean is set to 0
            self.velocities = np.random.normal(0, sigma, [self.N_particles, self.dimension])

            """
            mean_v, mean_v2 = [], []
            v2 = np.power(self.velocities, 2)
            for i in range(self.dimension):
                mean_v.append(np.mean(self.velocities[:, i]))
                mean_v2.append(np.mean(v2[:, i]))
            mean_v, mean_v2 = np.array(mean_v), np.array(mean_v2)
            # scale factor of the velocities
            f = np.sqrt(self.dimension * self.t_ref / mean_v2)
            # (self.velocities - mean_v): set initial momentum to 0
            # multiply by f: set initial kinetic energy to 1.5kbT (in 3D) or kbT (in 2D)
            self.velocities = (self.velocities - mean_v) * f
            """

        elif self.velo_method == 'debug':
            self.velocities = np.ones([self.N_particles, self.dimension])

        else:
            print('Error: The method for initializing the velocities should be either "random" or "maxwell".')
            sys.exit()

    def calc_dist(self, coord_i, coord_j):
        if self.PBC == 'no':
            dist = norm(coord_i - coord_j)
        elif self.PBC == 'yes':
            r_ij = coord_i - coord_j
            r_ij = r_ij - self.box_length * np.round(r_ij / self.box_length)
            dist = norm(r_ij)

        return dist


class ComputeForces(Initialization):
    def __init__(self, param):
        Initialization.__init__(self, param)

    def external_force_i(self, coord_i):
        """
        This function calculates the external force of the particle i given the 
        coordinates of particle i, which is np.array([-unxri^{n-2}, -unyri^{n-2}]). 
        """
        f_ext = np.zeros(self.dimension)  # x and y (and z) components
        origin = np.zeros(self.dimension)
        r_i = Initialization.calc_dist(self, coord_i, origin)
        f_ext[:] = - self.u * self.n * coord_i[:] * r_i ** (self.n - 2)

        return f_ext

    def interaction_force_ij(self, coord_i, coord_j):
        """
        This function calculates the interaction force between the particle i and 
        the particl j given the coordinates of the two particles, which is 
        np.array([ak(xi-xj)rij^{-k-2}, ak(yi-yj)rij^{-k-2}])
        """
        f_int = np.zeros(self.dimension)  # x and y (and z) components
        r_ij = Initialization.calc_dist(self, coord_i, coord_j)

        # Here we define that r_0 means no energy trunaction is used
        if (self.energy_truncation == 'yes' and r_ij < self.r_c) or (self.energy_truncation == 'no'):
            f_int[:] = self.a * self.k * \
                (coord_i - coord_j) * r_ij ** (-self.k - 2)
        else:
            pass  # so f_int = np.zeros(self.dimension)
        #f_int[:] = self.a * self.k * (coord_i - coord_j) * r_ij ** (-self.k - 2)

        return f_int

    def LJ_force_ij(self, coord_i, coord_j):
        # return f_LJ, a matrix with each component of the force
        f_LJ = np.zeros(self.dimension)  # x and y (and z) components
        r_ij = Initialization.calc_dist(self, coord_i, coord_j)

        # Here we define that r_0 means no energy trunaction is used
        if (self.energy_truncation == 'yes' and r_ij < self.r_c) or (self.energy_truncation == 'no'):
            r12 = (self.epsilon / r_ij) ** 12
            r6 = (self.epsilon / r_ij) ** 6
            r_ij_vec = (coord_i - coord_j)
            if self.PBC == 'yes':
                r_ij_vec -= self.box_length * np.round(r_ij_vec / self.box_length)
            f_LJ[:] = r_ij_vec * (48 * self.epsilon /
                                  (r_ij ** 2)) * (r12 - 0.5 * r6)

        else:
            pass  # so f_LJ = np.zeros(self.dimension)

        return f_LJ

    def total_force(self, coords, prtcl_list=None):
        """
        This function calculates the total force of all the particles given the coordinates
        of all the particles, which is an array of f_ext_i + sum_{j=i+1}^{N}(f_int_ij), where 
        i ranges from 1 to the number of particles
        """
        # Note that there is no need to apply PBC here. PBC will be applied outside the function
        # whenever it is needed. For example, in verlet integration, coords are the same for the last
        # step of n intergration and the first step in (n+1) integration. --> no need to apply PBC for both.

        if self.potential == 'central':
            # Step 1: First calculate the interaction force of each pair and store in a dictionary
            f0_int_dict = {}
            ij_pair = combinations(np.arange(1, self.N_particles + 1), 2)
            # ex. list(ij_pair) = [(1, 2), (1, 3), (2, 3)] if self.N_particles = 3
            for p in ij_pair:
                f0_int_dict[p] = self.interaction_force_ij(coords[p[0] - 1], coords[p[1] - 1])

            # Step 2: Complete the dictionary. Note that the f_int(i,j) = -f_int(j,i)
            f_int_dict = copy.deepcopy(f0_int_dict)
            for i in f0_int_dict:
                # ex. f_int_dict[2, 1] = -f0_int_dict[1, 2]
                f_int_dict[i[::-1]] = -f0_int_dict[i]

            # Step 3: Calculate the total interaction force of the particle i
            # (sum_{j=i+1}^{N}(f_int_ij)) and loop over i
            f_int_total = np.zeros([self.N_particles, self.dimension])
            for i in np.arange(self.N_particles):
                # [k for k in f_int_dict if k[0] == i] is [(1, 2), (1, 3)] if i = 1 for 3 particles
                for pair in [k for k in f_int_dict if k[0] == i + 1]:
                    f_int_total[i][:] += f_int_dict[pair][:]

            # Step 4: Calculate the total force for particle of the particle i and loop over i
            f_ext = np.zeros([self.N_particles, self.dimension])
            for i in range(self.N_particles):
                f_ext[i][:] = self.external_force_i(coords[i])

            f_total = f_int_total + f_ext
        
        if self.potential == 'LJ' or self.potential == 'WCA':
            f_matrix = np.zeros([self.dimension, self.N_particles, self.N_particles])
            # f_matrix[k, i, j] = k compoent of interaction acting on particle i by particle j
            # Note that the diagonal (fii) should be 0, so we'll just leave it there
            # Here we first calculate half of the matrix (j < i) for each k, since the matrix is symmetric
            for i in range(self.N_particles):

                if self.search_method == 'all-pairs':
                    loop_range = range(self.N_particles)
                elif self.search_method == 'verlet' or 'cell':
                    loop_range = prtcl_list[i]

                for j in loop_range:
                    if self.search_method == 'all-pairs' or self.search_method == 'verlet':
                        if j < i:
                            f_matrix[:, i, j] = self.LJ_force_ij(coords[i], coords[j])
                        
                        elif j > i:
                            break   # break the loop once cross the diagonal to prevent repeptive calculation

                    if self.search_method == 'cell':
                        f_matrix[:, i, j] = self.LJ_force_ij(coords[i], coords[j])
                        # since only half of the neighbors were considered in neighbor_cells
                        # looping over the whole cell list should give half (instead of the whole)
                        # f_matrix. Note that for vlist = [[0, 4, 5], ...], particle 0 must apppear
                        # in the list of partcile 4. However, if clist = [0, 4, 5], particle 0 should not 
                        # appear in the list of particle 4. This is the difference between clist and vlist.
            
            # then we finish the contruction of the symmetric matrix for each component
            for i in range(self.dimension):
                f_matrix[i] = -(f_matrix[i] -  f_matrix[i].transpose())

            # At last, calculate f_total for particle i
            f_total = np.zeros([self.N_particles, self.dimension])
            for i in range(self.dimension):
                f_total[:, i] = sum(f_matrix[i])

        else:
            print('Error: The type of the potential model should be either "central" or "LJ".')
            sys.exit()

        return f_total

    def angular_momentum(self, velocities, coords):
        """
        This function calculates the total magnitude of the angular momenta of the system.
        In this case, when r and p are both 2-D numpy array, magnitude of L = norm(np.cross(r, p))
        """
        # Note that PBC will be applied outside the function
        L_vec = np.cross(coords, velocities)
        L_total = 0
        for i in range(self.N_particles):
            L_total += L_vec[i]

        return norm(L_total)

    def virial(self, coords, prtcl_list=None):
        virial = 0
        for i in range(self.N_particles):

            if self.search_method == 'all-pairs':
                loop_range = range(self.N_particles)
            elif self.search_method == 'verlet':
                loop_range = prtcl_list[i]
            
            for j in loop_range:
                if j < i:
                    r_ij_vec = coords[i] - coords[j]
                    if self.PBC == 'yes':
                        r_ij_vec -= self.box_length * np.round(r_ij_vec / self.box_length)
                    f_ij_vec = self.LJ_force_ij(coords[i], coords[j])
                    virial += np.dot(f_ij_vec, r_ij_vec)

        virial *= (1/self.dimension)

        return virial

    def pressure(self, virial):
        # First calculate tail correction if needed
        p_tail = 0
        beta = 1 / self.t_ref
        if self.tail_correction == 'yes':
            rc_term = (2/3) * (1 / self.r_c) ** 9 - (1 / self.r_c) ** 3
            p_tail = (16/3) * np.pi * self.rho ** 2 * rc_term

        pressure = self.rho / beta + virial / \
            (self.box_length) ** 3 + p_tail    # p_tail could be 0

        return pressure


class ComputePotentials(Initialization):
    def __init__(self, param):
        Initialization.__init__(self, param)

    def external_potential_i(self, coord_i):
        origin = np.zeros(self.dimension)
        r_i = Initialization.calc_dist(self, coord_i, origin)
        p_ext = self.u * r_i ** self.n

        return p_ext

    def interaction_potential(self, coord_i, coord_j):
        r_ij = Initialization.calc_dist(self, coord_i, coord_j)

        # Here we define that r_0 means no energy trunaction is used
        if (self.energy_truncation == 'yes' and r_ij < self.r_c) or (self.energy_truncation == 'no'):
            p_int = self.a * r_ij ** (-self.k)
            if self.shift_energy == 'yes':
                pc_int = self.a * self.r_c ** (-self.k)
                p_int -= pc_int
        else:
            p_int = 0

        return p_int

    def LJ_potential(self, coord_i, coord_j):
        r_ij = Initialization.calc_dist(self, coord_i, coord_j)

        if (self.energy_truncation == 'yes' and r_ij < self.r_c) or (self.energy_truncation == 'no'):
            r12 = (self.epsilon / r_ij) ** 12
            r6 = (self.epsilon / r_ij) ** 6
            p_LJ = 4 * self.epsilon * (r12 - r6)

            if self.shift_energy == 'yes':
                rc = self.r_c - self.box_length * \
                    np.round(self.r_c / self.box_length)
                rc12 = (self.epsilon / rc) ** 12
                rc6 = (self.epsilon / rc) ** 6
                pc_LJ = 4 * self.epsilon * (rc12 - rc6)
                p_LJ -= pc_LJ
        else:
            p_LJ = 0

        return p_LJ

    def total_potential(self, coords, prtcl_list=None):
        # Note that PBC will be applied to coords before coords is input to this method.

        if self.potential == 'central':
            # Step 1: calculate \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} ar_{ij}^{-k}
            # Note that the total interaction energy is just the sum of the interaction
            # energy of all the pairs. There is only one energy between one pair.
            if self.potential == 'central':
                ij_pair = combinations(np.arange(1, self.N_particles + 1), 2)
                p_int_total = 0
                for p in ij_pair:
                    p_int_total += self.interaction_potential(
                        coords[p[0] - 1], coords[p[1] - 1])

                # Step 2: calculate \sum_{i=1}^{N}u*r_{i}^{N}
                p_ext = 0
                for i in range(self.N_particles):
                    p_ext += self.external_potential_i(coords[i])

                # Step 3: calculate total potential
                p_total = p_int_total + p_ext

        elif self.potential == 'LJ' or self.potential == 'WCA':
            # note that combinations returns a generator which produces its values when needed
            # instead of calculating everything at once and storing the result in memory.
            # Therefore, we don't use ij_pair = list(combinations(...)) and loop over the list, which is slower.
            p_total = 0
            #print(prtcl_list)  # worthy to take a look
            for i in range(self.N_particles):
                
                if self.search_method == 'all-pairs':
                    loop_range = range(self.N_particles)
                elif self.search_method == 'verlet':
                    loop_range = prtcl_list[i]
                elif self.search_method == 'cell':
                    loop_range = prtcl_list[i]
                
                for j in loop_range:
                    
                    if self.search_method == 'all-pairs' or self.search_method == 'verlet':
                        if j < i:
                            #print([i, j, self.LJ_potential(coords[i], coords[j])])
                            p_total += self.LJ_potential(coords[i], coords[j])
                        elif j > i:
                            break
                    
                    
                    if self.search_method == 'cell':
                        #if j != i:
                        #print([i, j, self.LJ_potential(coords[i], coords[j])])
                        p_total += self.LJ_potential(coords[i], coords[j])
                    
                    """
                    if self.search_method == 'cell':
                            p_total += self.LJ_potential(coords[i], coords[j])
                    """
            if self.tail_correction == 'yes':
                rc_term = (1 / 3) * (1 / self.r_c) ** 9 - (1 / self.r_c) ** 3
                u_tail = (8 / 3) * np.pi * self.rho * rc_term
                p_total += u_tail

        else:
            print(
                'Error: The type of the potential model should be either "central" or "LJ".')
            sys.exit()

        return p_total

    def LJ_pair_total(self, coords, i):
        # i: the index of the selected particle
        # useful for MC simulations
        E_pair_total = 0

        for j in range(self.N_particles):
            if i != j:
                E_pair_total += self.LJ_potential(coords[i], coords[j])

        return E_pair_total


class ParticleList(Initialization):
    def __init__(self, param):
        Initialization.__init__(self, param)

    def verlet_list(self, coords):
        self.r_v = self.delta + self.r_c
        v_list = []
        for i in range(self.N_particles):
            v_list.append([])  # initiliaze an empty list for each particle

        for i in range(self.N_particles - 1):
            for j in range(i + 1, self.N_particles):
                r_ij = self.calc_dist(coords[i], coords[j]) # PBC considered in calc_dist
                if r_ij < self.r_v:
                    v_list[i].append(j)
                    v_list[j].append(i)

        return v_list

    def cell_list(self, coords):
        # Given the coordinates of particles, this method identify the interacting 
        # particles in the same cell or  half of the neighboring cells defined in 
        # neighbors_cells. A cell list like [[2, 3, 7, 18], [5, 6, 4], ...] means
        # that partcile 0 interacts with particle 2, 3, 7, 10, which are either 
        # in the same or the neighbor cells of the cell  that particle 0 belongs to.
        # Interacting particles of particle i do not include the particle i itself.
        
        c_list = []
        for i in range(self.N_particles):
            c_list.append([])

        #cells, particles = self.all_cells(coords)
        self.all_cells(coords)   # so now we have self.cells and self.particles
        for i in range(self.N_particles):
            # Step 1: Given a particle, find the cell it belongs to 
            pos = self.particles[i]
            
            # Step 2: Identify the neighboring cells
            nn_cells = self.neighbor_cells(pos)
                        
            # Setp 3: Find the indices of the neighboring particles to loop over (not necessarily interacting)
            for j in range(len(nn_cells)):
                if j == 0:  # the same cell of particle i
                    c_list[i] += [k for k in self.cells[nn_cells[j]] if k > i]
                else:
                    c_list[i] = c_list[i] + self.cells[nn_cells[j]] 

        return c_list

    def find_cell(self, coord_i):
        # Given the coordinates of a particle, this function returns the cell (tuple) it belongs to
        coord_min = np.ones(self.dimension) * (-self.box_length * 0.5)  # lower left corner (cell (0, 0))
        pos = []
        r_ij_vec = coord_i - coord_min
        if self.PBC == 'yes':
                r_ij_vec -= self.box_length * np.round(r_ij_vec / self.box_length)
        for i in range(self.dimension):
            pos.append(int(np.floor(r_ij_vec[i] / self.r_cell)) % self.n_cell)
        pos = tuple(pos)

        return pos

    def all_cells(self, coords):

        # This function sort all the particles into different cells
        cell_pos = list(product(range(self.n_cell), repeat=self.dimension))
        self.cells, self.particles = {}, {}
        for i in range(self.n_cell ** 2):
            self.cells[cell_pos[i]] = []   # initilize the cell dictionary
            # In the end, cells = {(0, 0):[3, 7, 15], (0, 1):[1, 4, 8], (0, 2):[2, 31], ...}
            # which means that particle 3, 7, 15 are in cell (0, 0)

        for i in range(self.N_particles):
            pos = self.find_cell(coords[i])
            self.cells[pos].append(i)
            self.particles[i] = pos

        #return cells, particles
    

    def neighbor_cells(self, cell):
        # This function returns a list of the indices (tuple) of the cell of interest
        # and half of its neighbor cells. For a shematic representation, please refer 
        # Figure 5.5 (a) in Allen & Tildesley. If the input is cell 8, the the output
        # will be 8, 13, 4, 9, 4. PBC is applied. The input should be a tuple.
        # Note that this is only for 2D cell lists.
        # Example, input: (x, y), output: [(x, y), (x, y + 1), (x+1, y), (x + 1, y+1), (x + 1, y - 1)]
        # if (x, y) is not on the edge of the box.
        nn_cells = [cell]
        c = list(cell)  # center cell / the cell of interest
        if self.PBC == 'yes':
            nn_cells.append(tuple([c[0], (c[1] + 1) % self.n_cell]))
            nn_cells.append(tuple([(c[0] + 1) % self.n_cell, c[1]]))
            nn_cells.append(tuple([(c[0] + 1) % self.n_cell, (c[1] + 1) % self.n_cell]))
            nn_cells.append(tuple([(c[0] + 1) % self.n_cell, (c[1] - 1) % self.n_cell]))

        else:
            if c[0] + 1 < self.n_cell:
                nn_cells.append(tuple([c[0] + 1, c[1]]))
                if c[1] + 1 < self.n_cell:
                    nn_cells.append(tuple([c[0] + 1, c[1] + 1]))
                if c[1] -1 >= 0:
                    nn_cells.append(tuple([c[0] + 1, c[1] - 1]))
            if c[1] + 1 < self.n_cell:
                nn_cells.append(tuple([c[0], c[1] + 1]))

        return nn_cells


class Thermostats(Initialization):
    def __init__(self, param):
        Initialization.__init__(self, param)

    def andersen(self):
        pass

    def nose_hoover(self):
        pass


class MonteCarlo(ComputeForces, ComputePotentials):
    def __init__(self, param_obj):
        attr_dict = vars(param_obj)
        for key in attr_dict:
            setattr(self, key, attr_dict[key])
        Initialization.init_coords(self)

    def metropolis_algrtm(self, coords):
        for file in os.listdir('.'):
            if file == self.traj_name:
                # make the the output file is newly made
                os.remove(self.traj_name)

        # quantities at t = 0
        output0 = self.output_data(0, coords)
        outfile = open(self.traj_name, 'a+')
        outfile.write('# Output data of MC simulation\n')
        yaml.dump(output0, outfile, default_flow_style=False)
        coords_list = coords.tolist()  # to prevent line breaks when printing to the file

        if self.print_coords != 'no':
            outfile.write(
                'x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
            outfile.write(
                'y-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
            if self.dimension == 3:
                outfile.write(
                    'z-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')

        n_accept = 0   # for calculating the average acceptance probability
        n_trials = 0   # for calculating the instantaneous acceptance probability

        self.p_acc = []

        for i in range(self.N_steps):
            n_trials += 1

            # pick a particle, decide the random displacement and proposed new coordinates
            # the index of the selected particle
            i_particle = np.random.randint(self.N_particles)

            # Note that E_current and E_proposed are not total energies of the system
            # but the total pair energy related to the selection particle, i_particle
            # coords: current coordinates
            E_current = self.LJ_pair_total(coords, i_particle)

            # Now propose the coordinates for the selected particle
            d = (2 * np.random.rand(3) - 1) * self.max_d
            coords_proposed = copy.deepcopy(coords)
            coords_proposed[i_particle] += d
            if self.PBC == 'yes':
                coords_proposed[i_particle] -= self.box_length * \
                    np.round(coords_proposed[i_particle] / self.box_length)

            # Caculate the acceptance ratio based on the energy difference
            E_proposed = self.LJ_pair_total(coords_proposed, i_particle)
            delta_E = E_proposed - E_current
            beta = 1 / self.t_ref

            # metropolis-hasting algorithm
            if delta_E < 0:
                n_accept += 1
                E_current += delta_E   # so now E_current = E_proposed
                coords[i_particle] += d
                if self.PBC == 'yes':
                    coords -= self.box_length * \
                        np.round(coords / self.box_length)
            else:
                # Note that we don't calculate p_acc outside the if statement
                # since -beta * delta_E could be very large, which can cause overflow in exp
                p_acc = np.exp(-beta * delta_E)
                if np.random.rand() < p_acc:
                    n_accept += 1
                    E_current += delta_E   # so now E_current = E_proposed
                    coords[i_particle] += d
                    if self.PBC == 'yes':
                        coords -= self.box_length * \
                            np.round(coords / self.box_length)
                else:
                    pass  # so the coordinates and the energy remain the same

            self.p_acc.append(n_accept / n_trials)

            if i % 100 == 1:   # calculate p_acc every 1000 steps
                acc_rate = n_accept / n_trials
                if acc_rate <= 0.2:
                    self.max_d *= 0.8
                elif acc_rate >= 0.5:
                    self.max_d *= 1.2

            # print the trajectory data
            output = self.output_data(i + 1, coords)
            if i % self.print_freq == self.print_freq - 1:
                outfile.write("\n")
                yaml.dump(output, outfile, default_flow_style=False)
                if self.print_coords != 'no':
                    coords_list = coords.tolist()  # to prevent line breaks when printing to the file
                    outfile.write(
                        'x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
                    outfile.write(
                        'y-coordinates: ' + str([coords_list[i][1] for i in range(len(coords_list))]) + '\n')
                    if self.dimension == 3:
                        outfile.write(
                            'z-coordinates: ' + str([coords_list[i][2] for i in range(len(coords_list))]) + '\n')

        outfile.close()
        # Note: self.p_acc[-1] = n_accept / self.N_steps --> average acceptance ratio

    def output_data(self, i, coords):
        if self.PBC == 'yes':
            coords -= self.box_length * np.round(coords / self.box_length)
        output = OrderedDict()
        output['Step'] = i
        if self.print_Etotal != 'no':
            output['E_total'] = float(
                self.total_potential(coords))  # E_total = E_p
        if self.print_pressure != 'no':
            virial = self.virial(coords)
            output['Pressure'] = float(self.pressure(virial))

        return output


class MolecularDynamics(ComputeForces, ComputePotentials, ParticleList):
    def __init__(self, param_obj):
        attr_dict = vars(param_obj)
        for key in attr_dict:
            setattr(self, key, attr_dict[key])
        Initialization.init_coords(self)
        Initialization.init_velo(self)

    def verlet_integration(self, velocities, coords):
        """
        This function performs verlet integration given the initial coordinates and the 
        velocities of the particles. The arrays of the coordinates and the velocities
        are updated every MD step. The data is written to a file based on the printing 
        frequency specified in the MD parameters file.
        """

        for file in os.listdir('.'):
            if file == self.traj_name:
                # make the the output file is newly made
                os.remove(self.traj_name)

        # initial coords are certainly in the box. No need to apply PBC
        # Initilization of the list
        if self.search_method == 'verlet':
            vlist = ParticleList.verlet_list(self, coords)
            coords_old = copy.deepcopy(coords)
        if self.search_method == 'cell':
            clist = ParticleList.cell_list(self, coords)

        # quantities at t = 0
        if self.search_method == 'all-pairs':
            output0 = self.output_data(0, velocities, coords)
        elif self.search_method == 'verlet':
            output0 = self.output_data(0, velocities, coords, vlist)
        elif self.search_method == 'cell':
            output0 = self.output_data(0, velocities, coords, clist)

        outfile = open(self.traj_name, 'a+', newline='')
        outfile.write('# Output data of MD simulation\n')
        yaml.dump(output0, outfile, default_flow_style=False)
        if self.print_coords != 'no':
            coords_list = coords.tolist()  # to prevent line breaks when printing to the file
            outfile.write(
                'x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
            outfile.write(
                'y-coordinates: ' + str([coords_list[i][1] for i in range(len(coords_list))]) + '\n')
            if self.dimension == 3:
                outfile.write(
                    'z-coordinates: ' + str([coords_list[i][2] for i in range(len(coords_list))]) + '\n')

        # start the verlet integration
        # Note that PBC has been applied in total_force
        for i in range(self.N_steps):
            if self.search_method == 'all-pairs':
                v_half = velocities + (self.dt / (2 * self.m)) * \
                    ComputeForces.total_force(self, coords)
                # coordinates updated (from t to t+dt)
                coords = coords + v_half * self.dt
                if self.PBC == 'yes':
                    coords -= self.box_length * np.round(coords / self.box_length)
                velocities = v_half + (self.dt / (2 * self.m)) * \
                    ComputeForces.total_force(self, coords)

                # note that self.total_force(coords) in the line above is the total force
                # at t+dt, since the coordiantes have been updated.\
            
            elif self.search_method == 'verlet':
                v_half = velocities + (self.dt / (2 * self.m)) * \
                    ComputeForces.total_force(self, coords, vlist)
                coords = coords + v_half * self.dt
                if self.PBC == 'yes':
                    coords -= self.box_length * np.round(coords / self.box_length)
                coords_new = copy.deepcopy(coords)

                # check if the verlet list need to be updated
                d_vec = coords_new - coords_old
                d_max = max([norm(d_vec[i]) for i in range(len(d_vec))])
                if d_max >= self.delta * 0.5:
                    # coords_old only have to be updated if the list is updated
                    # so that we can accumulate the displacement 
                    vlist = ParticleList.verlet_list(self, coords)
                    coords_old = copy.deepcopy(coords)

                velocities = v_half + (self.dt / (2 * self.m)) * \
                    ComputeForces.total_force(self, coords, vlist)

            elif self.search_method == 'cell':
                v_half = velocities +(self.dt / (2 * self.m)) * \
                    ComputeForces.total_force(self, coords, clist)
                coords = coords + v_half * self.dt
                if self.PBC == 'yes':
                    coords -= self.box_length * np.round(coords / self.box_length)

                # check if the cell list needs to be update
                for i in range(self.N_particles):
                    pos = ParticleList.find_cell(self, coords[i])
                    if pos == self.particles[i]:
                        continue
                    else:
                        clist = ParticleList.cell_list(self, coords)
                        break


                #clist = ParticleList.cell_list(self, coords)
                velocities = v_half + (self.dt / (2 * self.m)) * \
                    ComputeForces.total_force(self, coords, clist)

            else:
                print('Error! The search method should be "all-pairs", "verlet", or "cell".')
                sys.exit()

            # Application of thermostat
            if self.tcoupl == 'Andersen' and i % self.nst_toupl - 1:
                n_selected = self.N_particles * self.mu * (self.nst_toupl * self.dt)    # the number of selected particles
                n_selected_int = int(np.floor(n_selected))
                # Note that n_selected could be a floating number. For example, if it is 4.3, we'll
                # select 4 particle first and there is a 30% chance that we select one more particle.
                
                selected_idx = np.random.randint(0, self.N_particles - 1, n_selected_int)
                # draw samples from np.arange(0, self.N_particles)
                selected_idx = np.random.choice(self.N_particles - 1, n_selected_int, replace=False)
                if n_selected - n_selected_int != 0:
                    if np.random.rand() < n_selected - n_selected_int:
                        candidates = list(np.arange(0, self.N_particles - 1))
                        for i in sorted(selected_idx, reverse=True):  # from big to small
                            del candidates[i]   # delete index i, that's why we sorted selected_idx
                        add_idx = np.random.choice(candidates, 1, replace=False)
                        selected_idx = np.concatenate((selected_idx, add_idx))

                sigma = np.sqrt(self.kb * self.t_ref / self.m)   # on the other hand, mean is set to 0
                for idx in range(len(selected_idx)):
                    # collision with the heat bath
                    velocities[[selected_idx]] = np.random.normal(0, sigma, [len(selected_idx), self.dimension])
                    # If A = array([49, 76, 55, 23, 31,  9, 35, 33,  0, 28]) and B = array([-6, -8])
                    # then after A[[3, 5]] = B, A becomes array([49, 76, 55, -6, 31, -8, 35, 33,  0, 28])
                    # This also applies for multi-dimensional arrays like we have here.

            if self.search_method == 'all-pairs':
                output = self.output_data(i + 1, velocities, coords)
            elif self.search_method == 'verlet':
                output = self.output_data(i + 1, velocities, coords, vlist)
            elif self.search_method == 'cell':
                output = self.output_data(i + 1, velocities, coords, clist)

            if i % self.print_freq == self.print_freq - 1:
                outfile.write("\n")
                yaml.dump(output, outfile, default_flow_style=False)
                if self.print_coords != 'no':
                    coords_list = coords.tolist()  # to prevent line breaks when printing to the file
                    outfile.write(
                        'x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
                    outfile.write(
                        'y-coordinates: ' + str([coords_list[i][1] for i in range(len(coords_list))]) + '\n')
                    if self.dimension == 3:
                        outfile.write(
                            'z-coordinates: ' + str([coords_list[i][2] for i in range(len(coords_list))]) + '\n')

    def output_data(self, i, velocities, coords, prtcl_list=None):
        output = OrderedDict()
        output['Step'] = i
        output['Time'] = self.dt * (i)

        # There are a lot of ways to calculate the square of the velocities:
        # Taking 2D velocities as an example:
        # 1. np.linalg.norm(v) (which is sqrt{vx1^2 + vy1^2 + vx2^2 + vy2^2 + ...})
        # 2. np.dot(v[:,0], v[:, 0]) + np.dot(v[:,1], v[:,1]) (for 2D)
        # 3. np.sum(np.diag(np.dot(v, v.transpose())))
        # speed: solution 2 > solution 1 >> solution 3.
        if self.print_Ek != 'no':
            output['E_k'] = float(0.5 * self.m * norm(velocities) ** 2)
        if self.print_Ep != 'no':
            output['E_p'] = float(self.total_potential(coords, prtcl_list))
        if self.print_Etotal != 'no' and self.print_Ek != 'no' and self.print_Ep != 'no':
            if self.print_Ek == 'yes' and self.print_Ep == 'yes':
                output['E_total'] = float(output['E_k'] + output['E_p'])
            else:
                output['E_total'] = float(0.5 * self.m * norm(velocities) ** 2) + float(self.total_potential(coords, prtcl_list))

        # Equilipartition theorem: <2Ek> = (3N-3)kT
        # Equil-partitiion theorem: <Ek> = (n/2) * kT, n = self.dimesion
        # so T = (2<Ek>)/(nk) = 2* sum(Ek) / (nNk)
        if self.print_temp != 'no':
            if self.print_Ek == 'yes':
                output['Temp'] = float(2 * output['E_k']/(self.dimension * self.N_particles * self.kb))
            else: 
                output['Temp'] = 2 * float(0.5 * self.m * norm(velocities) ** 2) / (self.dimension * self.N_particles * self.kb)

        if self.print_pressure != 'no':
            virial = self.virial(coords, prtcl_list)
            output['Pressure'] = float(self.pressure(virial))

        if self.print_L_total != 'no':
            output['L_total'] = float(
                self.angular_momentum(velocities, coords))

        return output


class TrajAnalysis:
    def __init__(self, param_obj, traj):
        np.set_printoptions(suppress=True)

        # copy the attributes instead of inheriting from Initialization (using Initialization.__init__(self, param))
        # so we can change the parameters externally by modifying the instance of Initialization
        attr_dict = vars(param_obj)
        for key in attr_dict:
            setattr(self, key, attr_dict[key])

        if self.simulation == 'MD':
            self.time, self.E_k, self.E_p, self.temp, self.L = [], [], [], [], []

        # The following are documented in traj.yml for both MC and MD
        self.step, self.E_total, self.pressure = [], [], []
        self.x = np.zeros([self.N_particles, int(
            np.ceil(self.N_steps / self.print_freq)) + 1])
        self.y = np.zeros([self.N_particles, int(
            np.ceil(self.N_steps / self.print_freq)) + 1])
        self.z = np.zeros([self.N_particles, int(
            np.ceil(self.N_steps / self.print_freq)) + 1])

        f = open(traj, 'r')
        lines = f.readlines()
        f.close()

        for l in lines:
            if 'Step: ' in l:
                self.step.append(int(l.split(':')[1]))
            if 'Time: ' in l:
                self.time.append(float(l.split(':')[1]))
            if 'E_k: ' in l:
                self.E_k.append(float(l.split(':')[1]))
            if 'E_p: ' in l:
                self.E_p.append(float(l.split(':')[1]))
            if 'E_total: ' in l:
                self.E_total.append(float(l.split(':')[1]))
            if 'Temp: ' in l:
                self.temp.append(float(l.split(':')[1]))
            if 'L_total: ' in l:
                self.L.append(float(l.split(':')[1]))
            if 'Pressure: ' in l:
                self.pressure.append(float(l.split(':')[1]))

            if 'x-coordinates: ' in l:
                for i in range(self.N_particles):
                    self.x[i][len(self.step) - 1] = float(l.split(':')
                                                          [1].split('[')[1].split(']')[0].split(',')[i])
            if 'y-coordinates: ' in l:
                for i in range(self.N_particles):
                    self.y[i][len(self.step) - 1] = float(l.split(':')
                                                          [1].split('[')[1].split(']')[0].split(',')[i])
            if 'z-coordinates: ' in l:
                # might not be used
                for i in range(self.N_particles):
                    self.z[i][len(self.step) - 1] = float(l.split(':')
                                                          [1].split('[')[1].split(']')[0].split(',')[i])

    def calculate_RMSF(self, y):
        # y: any quantity
        y_avg = np.mean(y)
        y2_avg = np.mean(np.power(y, 2))
        RMSF = np.sqrt((y2_avg - y_avg ** 2)) / y_avg

        return RMSF

    def plot_SMA(self, n, y, y_name, y_unit=None):
        # SMA: simple moving average (for time series)
        # n: the size of the subset
        n_points = len(y) - n + 1   # number of points
        SMA = []
        for i in range(n_points):
            SMA.append(np.mean(np.array(y[i:i + n])))
        step = np.arange(0, self.N_steps + 1, self.print_freq)[n - 1:]
        plt.title('SMA of %s as a function of simulation step' % y_name)
        plt.plot(step, SMA)
        plt.xlabel('Simulation step')
        if y_unit is None:
            plt.ylabel('%s ' % y_name)
        else:
            plt.ylabel('%s (%s)' % (y_name, y_unit))
        plt.grid()
        print('The average of the last subset: %s' %SMA[-1])

    def plot_2d(self, y, y_name, truncate=0, y_unit=None):
        plt.figure()
        x = np.arange(0, self.N_steps + 1, self.print_freq)[truncate:]
        y = y[truncate:]
        plt.plot(x, y)
        plt.title('%s as a function of simulation step' % y_name)
        plt.xlabel('Simulation step')
        if y_unit is None:
            plt.ylabel('%s ' % y_name)
        else:
            plt.ylabel('%s (%s)' % (y_name, y_unit))
        plt.grid()

    def plot_MD_energy(self, truncate=0):
        plt.figure()
        x = np.arange(0, self.N_steps + 1, self.print_freq)[truncate:]
        plt.plot(x[truncate:], np.array(self.E_k[truncate:]) /
                 self.N_particles, label='Kinetic energy')
        plt.plot(x[truncate:], np.array(self.E_p[truncate:]) /
                 self.N_particles, label='Potential energy')
        plt.plot(x[truncate:], np.array(self.E_total[truncate:]) /
                 self.N_particles, label='Total energy')
        plt.xlabel('Timestep')
        plt.ylabel('Energy per particle')
        plt.legend()
        plt.grid()

    def plot_xy_traj(self, truncate=0):
        plt.figure()
        for i in range(self.N_particles):
            plt.scatter(self.x[i][truncate:], self.y[i][truncate:], c=plt.cm.GnBu(
                np.linspace(0, 1, len(self.step))[truncate:]))
        plt.title('Trajectory of the particles in the x-y plane')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.grid()

    def plot_all_MD(self):
        self.plot_2d(self.E_k, 'Kinetic energy')
        self.plot_2d(self.E_p, 'Potential energy')
        self.plot_2d(self.E_total, 'Total energy')
        self.plot_2d(self.temp, 'Temperature')
        self.plot_2d(self.L, 'Angular momentum')
        self.plot_xy_traj()
