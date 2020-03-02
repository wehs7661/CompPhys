import os
import sys
import yaml
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from numpy.linalg import norm
from collections import OrderedDict
from itertools import combinations
from itertools import product
from matplotlib import rc

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
        with open(param) as ymlfile:
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
        if self.simulation == 'MC' or (self.simulation == 'MD' and self.velo_method == 'temp_rescale'):
            # temeprature must be specified in an NVT MC simulation or if temp_rescale is used
            self.examine_params('temperature', True)

        if self.simulation == 'MD':
            if 'temperature' not in self.param:
                self.velo_method = 'random'
            self.examine_params('velo_method', False, 'random')

        if 'box_length' not in self.param and 'rho' not in self.param:
            print('Error: At least one of the the length of the box or particle density should be specified.')
            sys.exit()

        if 'box_length' not in self.param and 'rho' in self.param:
            # only if box_legth is not specificed is rho activated
            self.box_length = (self.N_particles / self.rho) ** (1 / 3)

        # assign defaults to non-specified paramters
        if self.simulation == 'MD':
            self.examine_params('dt', False, 0.01)
        
        if self.potential == 'LJ':
            self.examine_params('epsilon', False, 1)
            self.examine_params('sigma', False, 1)

        self.examine_params('print_freq', False, 1)
        self.examine_params('dimension', False, 3)
        self.examine_params('coords_method', False, 'random')
        self.examine_params('PBC', False, 'yes')
        self.examine_params('m', False, 1)
        self.examine_params('kb', False, 1)
        self.examine_params('energy_truncation', False, 'no')
        self.examine_params('shift_energy', False, 'no')

        if self.energy_truncation == 'yes':
            self.examine_params('r_c', False, 0.5 * self.box_length)

        # Step 2: Print the adopted parameters to a yaml file
        adopted_params = copy.deepcopy(vars(self))
        del adopted_params['param']
        out_param_name = param.split('.')[0] + '_adopted_params.yml'

        for file in os.listdir('.'):
            if file == out_param_name:
                # make the the output file is newly made
                os.remove(out_param_name)

        with open(out_param_name, 'a+', newline='') as out_params:
            out_params.write('# Adopted simulation parameters\n')
            yaml.dump(adopted_params, out_params, default_flow_style=False)

        # Step 3: Add other attributes
        self.prefix = param.split('.')[0]
        self.traj_name = self.prefix + '_traj.yml'

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
            self.coords = (0.5 - np.random.rand(self.N_particles, self.dimension)) * self.box_length  # initial coordinates
        elif self.coords_method == 'lattice':
            N = np.ceil((self.N_particles)**(1 / self.dimension))      # number of grids per side of the lattice
            r_min = -self.box_length / 2
            r_max = self.box_length / 2
            self.d = self.box_length / N   # initial spacing between particles
            pos = np.linspace(r_min + 0.5 * self.d, r_max - 0.5 * self.d, int(N))
            coords_list = list(product(pos, repeat=self.dimension))    # length is larger or equil to self.coords     
            self.coords = np.zeros([self.N_particles, self.dimension])
            for i in range(len(self.coords)):
                self.coords[i] = list(coords_list[i])
        else:
            print('Error: The method for initializing the coordinates should be either "random" or "lattice".')
            sys.exit()
    
    def init_velo(self):
        if self.velo_method == 'random':
            self.velocities = np.random.rand(
                self.N_particles, self.dimension) * self.box_length * 0.1  # initial velocity
        
        elif self.velo_method == 'temp_rescale':
            # reference: http://www.cchem.berkeley.edu/chem195/_l_j___andersen_thermostat_8m.html
            # set initial velocities to random number
            sigma = np.sqrt(self.temperature / self.m)
            self.velocities = np.random.normal(scale=sigma, size=[self.N_particles, self.dimension])

            mean_v, mean_v2 = [], []
            v2 = np.power(self.velocities, 2)
            for i in range(self.dimension):
                mean_v.append(np.mean(self.velocities[:, i]))
                mean_v2.append(np.mean(v2[:, i]))
            mean_v, mean_v2 = np.array(mean_v), np.array(mean_v2)
            f = np.sqrt(self.dimension * self.temperature / mean_v2 )        # scale factor of the velocities
            # (self.velocities - mean_v): set initial momentum to 0
            # multiply by f: set initial kinetic energy to 1.5kbT (in 3D)
            self.velocities = (self.velocities - mean_v) * f
            self.velocities = np.random.normal(scale=sigma, size=[self.N_particles, self.dimension])

        else:
            print('Error: The method for initializing the velocities should be either "random" or "temp_rescale".')
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
        f_ext = np.zeros([1, self.dimension])  # x and y (and z) components
        origin = np.zeros([1, self.dimension])
        r_i = Initialization.calc_dist(self, coord_i, origin)
        f_ext[:] = - self.u * self.n * coord_i[:] * r_i ** (self.n - 2)

        return f_ext

    def interaction_force_ij(self, coord_i, coord_j):
        """
        This function calculates the interaction force between the particle i and 
        the particl j given the coordinates of the two particles, which is 
        np.array([ak(xi-xj)rij^{-k-2}, ak(yi-yj)rij^{-k-2}])
        """
        f_int = np.zeros([1, self.dimension])  # x and y (and z) components
        r_ij = Initialization.calc_dist(self, coord_i, coord_j)

        # Here we define that r_0 means no energy trunaction is used
        if (self.energy_truncation == 'yes' and r_ij < self.r_c) or (self.energy_truncation == 'no'):
            f_int[:] = self.a * self.k * (coord_i - coord_j) * r_ij ** (-self.k - 2)
        else:
            pass # so f_int = np.zeros([1, self.dimension])
        f_int[:] = self.a * self.k * (coord_i - coord_j) * r_ij ** (-self.k - 2)

        return f_int

    def LJ_force_ij(self, coord_i, coord_j):
        f_LJ = np.zeros([1, self.dimension])  # x and y (and z) components
        r_ij = Initialization.calc_dist(self, coord_i, coord_j)

        # Here we define that r_0 means no energy trunaction is used
        if (self.energy_truncation == 'yes' and r_ij < self.r_c) or (self.energy_truncation == 'no'):
            r12 = (self.epsilon / r_ij) ** 12
            r6 = (self.epsilon / r_ij) ** 6
            rij_vec = (coord_i - coord_j) 
            rij_vec -= self.box_length * np.round(rij_vec / self.box_length)
            f_LJ[:] = rij_vec * (48 * self.epsilon / (r_ij ** 2)) * (r12 - 0.5 * r6)
        
        else:
            pass  # so f_LJ = np.zeros([1, self.dimension])

        return f_LJ

    def total_force(self, coords):
        """
        This function calculates the total force of all the particles given the coordinates
        of all the particles, which is an array of f_ext_i + sum_{j=i+1}^{N}(f_int_ij), where 
        i ranges from 1 to the number of particles
        """
        if self.PBC == 'yes':
            coords -= self.box_length * np.round(coords / self.box_length)

        if self.potential == 'central':
            # Step 1: First calculate the interaction force of each pair and store in a dictionary
            f0_int_dict = {}
            ij_pair = list(combinations(np.arange(1, self.N_particles + 1), 2))
            # ex. ij_pair = [(1, 2), (1, 3), (2, 3)] if self.N_particles = 3
            for p in ij_pair:
                f0_int_dict[p] = self.interaction_force_ij(
                    coords[p[0] - 1], coords[p[1] - 1])

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
                    f_int_total[i][:] += f_int_dict[pair][0][:]

            # Step 4: Calculate the total force for particle of the particle i and loop over i
            f_ext = np.zeros([self.N_particles, self.dimension])
            for i in range(self.N_particles):
                f_ext[i][:] = self.external_force_i(coords[i])

            f_total = f_int_total + f_ext

        elif self.potential == 'LJ':
            f0_LJ_dict = {}
            ij_pair = list(combinations(np.arange(1, self.N_particles + 1), 2))
            for p in ij_pair:
                f0_LJ_dict[p] = self.LJ_force_ij(coords[p[0] - 1], coords[p[1] - 1])

            f_LJ_dict = copy.deepcopy(f0_LJ_dict)
            for i in f0_LJ_dict:
                f_LJ_dict[i[::-1]] = -f0_LJ_dict[i]

            f_total = np.zeros([self.N_particles, self.dimension])
            for i in np.arange(self.N_particles):
                for pair in [k for k in f_LJ_dict if k[0] == i + 1]:
                    f_total[i][:] += f_LJ_dict[pair][0][:]
    
        else:
            print('Error: The type of the potential model should be either "central" or "LJ".')
            sys.exit()

        return f_total

    def angular_momentum(self, velocities, coords):
        """
        This function calculates the total magnitude of the angular momenta of the system.
        In this case, when r and p are both 2-D numpy array, magnitude of L = norm(np.cross(r, p))
        """
        if self.PBC == 'yes':
            coords -= self.box_length * np.round(coords / self.box_length)
        L_vec = np.cross(coords, velocities)
        L_total = 0
        for i in range(self.N_particles):
            L_total += L_vec[i]

        return norm(L_total)


class ComputePotentials(Initialization):
    def __init__(self, param):
        Initialization.__init__(self, param)

    def external_potential_i(self, coord_i):
        origin = np.zeros([1, self.dimension])
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
                rc12 = (self.epsilon / self.r_c) ** 12
                rc6 = (self.epsilon / self.r_c) ** 6
                pc_LJ = 4 * self.epsilon * (rc12 - rc6)
                p_LJ -= pc_LJ
        else:
            p_LJ = 0

        return p_LJ

    def total_potential(self, coords):
        if self.PBC == 'yes':
            coords -= self.box_length * np.round(coords / self.box_length)

        if self.potential == 'central':
            # Step 1: calculate \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} ar_{ij}^{-k}
            # Note that the total interaction energy is just the sum of the interaction
            # energy of all the pairs. There is only one energy between one pair.
            if self.potential == 'central':
                ij_pair = list(combinations(
                    np.arange(1, self.N_particles + 1), 2))
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

        elif self.potential == 'LJ':
            ij_pair = list(combinations(
                    np.arange(1, self.N_particles + 1), 2))
            p_total = 0
            for p in ij_pair:
                p_total += self.LJ_potential(coords[p[0] - 1], coords[p[1] - 1])

        else:
            print('Error: The type of the potential model should be either "central" or "LJ".')
            sys.exit()

        return p_total

class MonteCarlo(ComputePotentials):
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
        with open(self.traj_name, 'a+', newline='') as outfile:
            outfile.write('# Output data of MC simulation\n')
            yaml.dump(output0, outfile, default_flow_style=False)
            coords_list = coords.tolist()  # to prevent line breaks when printing to the file
            outfile.write('x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
            outfile.write('y-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
            if self.dimension == 3:
                outfile.write('z-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')

        n_accept = 0   # for calculating the average acceptace probability

        for i in range(self.N_steps):
            E_current = self.total_potential(coords)  # coords: current coordinates

            # pick a particle, decide the random displacement and proposed new coordinates
            i_particle = np.random.randint(self.N_particles)  # selected particle
            d = (2 *np.random.rand(3) - 1) * self.max_d
            coords_proposed = copy.deepcopy(coords)
            coords_proposed[i_particle] += d
            if self.PBC == 'yes':
                coords_proposed -= self.box_length * np.round(coords_proposed / self.box_length)
            
            # Caculate the acceptance ratio based on the energy difference
            E_proposed = self.total_potential(coords_proposed)
            delta_E = E_proposed - E_current
            beta = 1 / self.temperature
            p_acc = np.exp(-beta * delta_E)

            # metropolis-hasting algorithm
            if delta_E < 0:
                n_accept += 1
                E_current += delta_E   # so now E_current = E_proposed
                coords[i_particle] += d
                if self.PBC == 'yes':
                    coords -= self.box_length * np.round(coords / self.box_length)
            else:
                if np.random.rand() < p_acc:
                    n_accept += 1
                    E_current += delta_E   # so now E_current = E_proposed
                    coords[i_particle] += d
                    if self.PBC == 'yes':
                        coords -= self.box_length * np.round(coords / self.box_length)
                else:
                    pass  # so the coordinates and the energy remain the same

            # print the trajectory data
            output = self.output_data(i + 1, coords)
            with open(self.traj_name, 'a+', newline='') as outfile:
                if i % self.print_freq == self.print_freq - 1:
                    outfile.write("\n")
                    yaml.dump(output, outfile, default_flow_style=False)
                    coords_list = coords.tolist()  # to prevent line breaks when printing to the file
                    outfile.write('x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
                    outfile.write('y-coordinates: ' + str([coords_list[i][1] for i in range(len(coords_list))]) + '\n')
                    if self.dimension == 3:
                        outfile.write('z-coordinates: ' + str([coords_list[i][2] for i in range(len(coords_list))]) + '\n')

        self.acc_rate = n_accept / self.N_steps

    def output_data(self, i, coords):
        if self.PBC == 'yes':
            coords -= self.box_length * np.round(coords / self.box_length)
        output = OrderedDict()
        output['Step'] = i
        output['E_total'] = float(self.total_potential(coords))  # E_total = E_p

        return output


class MolecularDynamics(ComputeForces, ComputePotentials):
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

        # quantities at t = 0
        output0 = self.output_data(0, velocities, coords)

        with open(self.traj_name, 'a+', newline='') as outfile:
            outfile.write('# Output data of MD simulation\n')
            yaml.dump(output0, outfile, default_flow_style=False)
            coords_list = coords.tolist()  # to prevent line breaks when printing to the file
            outfile.write('x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
            outfile.write('y-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
            if self.dimension == 3:
                outfile.write('z-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')

        for i in range(self.N_steps):
            if self.PBC == 'yes':
                coords -= self.box_length * np.round(coords / self.box_length)
            v_half = velocities + (self.dt / (2 * self.m)) * \
                ComputeForces.total_force(self, coords)
            coords = coords + v_half * self.dt  # coordinates updated (from t to t+dt)
            if self.PBC == 'yes':
                coords -= self.box_length * np.round(coords / self.box_length)
            velocities = v_half + (self.dt / (2 * self.m)) * \
                ComputeForces.total_force(self, coords)
            
            # note that self.total_force(coords) in the line above is the total force
            # at t+dt, since the coordiantes have been updated.\
            
            output = self.output_data(i + 1, velocities, coords)
            with open(self.traj_name, 'a+', newline='') as outfile:
                if i % self.print_freq == self.print_freq - 1:
                    outfile.write("\n")
                    yaml.dump(output, outfile, default_flow_style=False)
                    coords_list = coords.tolist()  # to prevent line breaks when printing to the file
                    outfile.write('x-coordinates: ' + str([coords_list[i][0] for i in range(len(coords_list))]) + '\n')
                    outfile.write('y-coordinates: ' + str([coords_list[i][1] for i in range(len(coords_list))]) + '\n')
                    if self.dimension == 3:
                        outfile.write('z-coordinates: ' + str([coords_list[i][2] for i in range(len(coords_list))]) + '\n')

    def output_data(self, i, velocities, coords):
        if self.PBC == 'yes':
            coords -= self.box_length * np.round(coords / self.box_length)
        output = OrderedDict()
        output['Step'] = i
        output['Time'] = self.dt * (i)
        # norm(v) = sqrt{vx1^2 + vy1^2 + vx2^2 + vy2^2 + ...}
        output['E_k'] = float(0.5 * self.m * norm(velocities) ** 2)
        output['E_p'] = float(self.total_potential(coords))
        output['E_total'] = float(output['E_k'] + output['E_p'])
        # Equilipartition theorem: <2Ek> = (3N-3)kT
        output['Temp'] = float((2 * output['E_k']) / ((3 * self.N_particles - 3) *self.kb))
        output['L_total'] = float(self.angular_momentum(velocities, coords))

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
        self.step, self.E_total = [], []
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
            
            if 'x-coordinates: ' in l:
                for i in range(self.N_particles):
                    self.x[i][len(self.step) - 1] = float(l.split(':')[1].split('[')[1].split(']')[0].split(',')[i])
            if 'y-coordinates: ' in l:
                for i in range(self.N_particles):
                    self.y[i][len(self.step) - 1] = float(l.split(':')
                                                          [1].split('[')[1].split(']')[0].split(',')[i])
            if 'z-coordinates: ' in l:
                # might not be used
                for i in range(self.N_particles):
                    self.z[i][len(self.step) - 1] = float(l.split(':')
                                                          [1].split('[')[1].split(']')[0].split(',')[i])

    def calculate_RMSF(self, E_total):
        E_avg = np.mean(E_total)
        E2_avg = np.mean(np.power(E_total, 2))
        RMSF = np.sqrt((E2_avg - E_avg ** 2)) / E_avg

        return RMSF
            

    def plot_2d(self, y, y_name, truncate=0, y_unit=None):
        plt.figure()
        x = np.arange(len(y))[truncate:] * self.print_freq
        y = y[truncate:]
        plt.plot(x, y)
        plt.title('%s as a function of simulation step' % y_name)
        plt.xlabel('Simulation step')
        if y_unit is None:
            plt.ylabel('%s ' % y_name)
        else:
            plt.ylabel('%s (%s)' % (y_name, y_unit))
        plt.grid()

    def plot_MD_energy(self):
        plt.figure()
        x = np.arange(len(y))* self.print_freq
        plt.plot(x, np.array(self.E_k) / self.N_particles, label='Kinetic energy')
        plt.plot(x, np.array(self.E_p) / self.N_particles, label='Potential energy')
        plt.plot(x, np.array(self.E_total) / self.N_particles, label='Total energy')
        plt.xlabel('Timestep')
        plt.ylabel('Energy per particle')
        plt.legend()
        plt.grid()

    def plot_xy_traj(self):
        plt.figure()
        for i in range(self.N_particles):
            plt.scatter(self.x[i], self.y[i], c=plt.cm.GnBu(
                np.linspace(0, 1, len(self.step))))
        plt.title('Trajectory of the particles in the x-y plane')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        if self.PBC == 'yes':
            plt.xlim([-0.5 * self.box_length, 0.5 * self.box_length])
            plt.ylim([-0.5 * self.box_length, 0.5 * self.box_length])
            plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()

    def plot_all_MD(self):
        self.plot_2d(self.E_k, 'Kinetic energy')
        self.plot_2d(self.E_p, 'Potential energy')
        self.plot_2d(self.E_total, 'Total energy')
        self.plot_2d(self.temp, 'Temperature')
        self.plot_2d(self.L, 'Angular momentum')
        self.plot_xy_traj()
