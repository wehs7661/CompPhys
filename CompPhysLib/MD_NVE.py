import os
import yaml
import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from collections import OrderedDict
from itertools import combinations 
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

class MDParameters:
    def __init__(self, mdp):                         
        with open(mdp) as ymlfile:
            self.mdp = yaml.load(ymlfile)
        
        for attr in self.mdp:
            setattr(self, attr, self.mdp[attr])
            
        self.kb = 1  # Boltzmann constant
        
class ComputeForces(MDParameters):
    def __init__(self, mdp):
        MDParameters.__init__(self, mdp)
    
    def external_force_i(self, coord_i):
        """
        This function calculates the external force of the particle i given the 
        coordinates of particle i, which is np.array([-unxri^{n-2}, -unyri^{n-2}]). 
        """
        f_ext = np.zeros([1, 2])  # x and y components
        r_i = norm(coord_i)
        f_ext[:] = - self.u * self.n * coord_i[:] * r_i ** (self.n - 2)
        
        return f_ext
    
    def interaction_force_ij(self, coord_i, coord_j):
        """
        This function calculates the interaction force between the particle i and 
        the particl j given the coordinates of the two particles, which is 
        np.array([ak(xi-xj)rij^{-k-2}, ak(yi-yj)rij^{-k-2}])
        """
        f_int = np.zeros([1, 2])  # x and y components
        r_ij = norm(coord_i - coord_j)
        f_int[:] = self.a * self.k * (coord_i - coord_j) * r_ij ** (-self.k - 2)
        
        return f_int
    
    def total_force(self, coords):
        """
        This function calculates the total force of all the particles given the coordinates
        of all the particles, which is an array of f_ext_i + sum_{j=i+1}^{N}(f_int_ij), where 
        i ranges from 1 to the number of particles
        """
        # Step 1: First calculate the interaction force of each pair and store in a dictionary
        f0_int_dict = {}
        ij_pair = list(combinations(np.arange(1, self.N_particles + 1), 2))
        # ex. ij_pair = [(1, 2), (1, 3), (2, 3)] if self.N_particles = 3
        for p in ij_pair:
            f0_int_dict[p] = self.interaction_force_ij(coords[p[0] - 1], coords[p[1] - 1])
        
        # Step 2: Complete the dictionary. Note that the f_int(i,j) = -f_int(j,i)
        f_int_dict = copy.deepcopy(f0_int_dict)
        for i in f0_int_dict:
            f_int_dict[i[::-1]] = -f0_int_dict[i]  # ex. f_int_dict[2, 1] = -f0_int_dict[1, 2]
        
        # Step 3: Calculate the total interaction force of the particle i 
        # (sum_{j=i+1}^{N}(f_int_ij)) and loop over i
        f_int_total = np.zeros([self.N_particles, 2])
        for i in np.arange(self.N_particles):
            # [k for k in f_int_dict if k[0] == i] is [(1, 2), (1, 3)] if i = 1 for 3 particles
            for pair in [k for k in f_int_dict if k[0] == i + 1]: 
                f_int_total[i][:] += f_int_dict[pair][0][:]
        
        # Step 4: Calculate the total force for particle of the particle i and loop over i
        f_ext = np.zeros([self.N_particles, 2])
        for i in range(self.N_particles):
            f_ext[i][:] = self.external_force_i(coords[i])
        
        f_total = f_int_total + f_ext
        
        return f_total
    
    def angular_momentum(self, velocities, coords):
        """
        This function calculates the total magnitude of the angular momenta of the system.
        In this case, when r and p are both 2-D numpy array, magnitude of L = norm(np.cross(r, p))
        """
        L_vec = np.cross(coords, velocities)
        L_total = 0
        for i in range(self.N_particles):
            L_total += L_vec[i]
            
        return L_total

    
class ComputePotentials(MDParameters):
    def __init__(self, mdp):
        MDParameters.__init__(self, mdp)
    
    def external_potential_i(self, coord_i):
        r_i = norm(coord_i)
        p_ext = self.u * r_i ** self.n
        
        return p_ext
    
    def interaction_potential(self, coord_i, coord_j):
        r_ij = norm(coord_i - coord_j)
        if self.r_c == 0:
            # Here we define that r_0 means no energy trunaction is used
            p_int = self.a * r_ij ** (-self.k)
        else:
            if r_ij < self.r_c:
                p_int = self.a * r_ij ** (-self.k)
            else:
                p_int = 0
        
        return p_int
    
    def total_potential(self, coords):
        # Step 1: calculate \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} ar_{ij}^{-k}
        # Note that the total interaction energy is just the sum of the interaction 
        # energy of all the pairs. There is only one energy between one pair.
        
        ij_pair = list(combinations(np.arange(1, self.N_particles + 1), 2))
        p_int_total = 0
        for p in ij_pair:
            p_int_total += self.interaction_potential(coords[p[0] - 1], coords[p[1] - 1])
            
        # Step 2: calculate \sum_{i=1}^{N}u*r_{i}^{N}
        p_ext = 0
        for i in range(self.N_particles):
            p_ext += self.external_potential_i(coords[i])
            
        # Step 3: calculate total potential
        p_total = p_int_total + p_ext
        
        return p_total
    
    
class MolecularDynamics(ComputeForces, ComputePotentials):
    def __init__(self, mdp):
        MDParameters.__init__(self, mdp)
        
        # initial conditions
        self.coords = (0.5 - np.random.rand(self.N_particles, 2)) * self.box_length  # initial coordinates
        self.velocities = np.random.rand(self.N_particles, 2) * self.box_length * 0.1   # initial velocity
    
    def output_data(self, i, velocities, coords):
        output = OrderedDict()
        output['Step'] = i
        output['Time'] = self.dt * (i)
        output['E_k'] = float(0.5 * self.m * norm(velocities) ** 2)   # norm(v) = sqrt{vx1^2 + vy1^2 + vx2^2 + vy2^2 + ...}
        output['E_p'] = float(self.total_potential(coords))
        output['E_total'] = float(output['E_k'] + output['E_p'])
        output['Temp'] = float((2 * output['E_k']) / (3 * self.kb))     # Equilipartition theorem: <Ek> = (3/2)kT
        output['L_total'] = float(self.angular_momentum(velocities, coords))
        #output['X-coordinates'] = np.float64(coords[0])
        
        return output
        
    def verlet_integration(self, velocities, coords):
        """
        This function performs verlet integration given the initial coordinates and the 
        velocities of the particles. The arrays of the coordinates and the velocities
        are updated every MD step. The data is written to a file based on the printing 
        frequency specified in the MD parameters file.
        """
 
        for file in os.listdir('.'):
            if file == 'MD_traj.yml':
                os.remove('MD_traj.yml')  # make the the output file is newly made
                
        # quantities at t = 0
        output0 = self.output_data(0, velocities, coords)
                
        with open('MD_traj.yml', 'a+') as outfile:
            outfile.write('# Output data of MD simulation\n') 
            yaml.dump(output0, outfile, default_flow_style=False)
            outfile.write('x-coordinates: ' + str(coords[:, 0]) + '\n')
            outfile.write('y-coordinates: ' + str(coords[:, 1]) + '\n')
            
            
        for i in range(self.N_steps):
            v_half = velocities + (self.dt / (2 * self.m)) * ComputeForces.total_force(self, coords)
            coords = coords + v_half * self.dt  # coordinates updated (from t to t+dt)
            velocities = v_half + (self.dt / (2 * self.m)) * ComputeForces.total_force(self, coords)
            # note that self.total_force(coords) in the line above is the total force
            # at t+dt, since the coordiantes have been updated.
            
            output = self.output_data(i + 1, velocities, coords)
            
            with open('MD_traj.yml', 'a+') as outfile:
                if i % self.print_freq == self.print_freq - 1:
                    outfile.write("\n")
                    yaml.dump(output, outfile, default_flow_style=False)
                    outfile.write('x-coordinates: ' + str(coords[:, 0]) + '\n')
                    outfile.write('y-coordinates: ' + str(coords[:, 1]) + '\n')
            

class MDAnalysis(MDParameters):
    def __init__(self, mdp, traj):
        np.set_printoptions(suppress=True)
        MDParameters.__init__(self, mdp)
        self.step, self.time, self.E_k, self.E_p = [], [], [], []
        self.E_total, self.temp, self.L = [], [], []
        self.x = np.zeros([self.N_particles, int(np.ceil(self.N_steps / self.print_freq)) + 1])
        self.y = np.zeros([self.N_particles, int(np.ceil(self.N_steps / self.print_freq)) + 1])
        
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
                    #print(self.time[-1])
                    #print(l.split(':')[1].split('[')[1].split(']')[0].split()[i])
                    self.x[i][len(self.time) - 1] = float(l.split(':')[1].split('[')[1].split(']')[0].split()[i])
            if 'y-coordinates: ' in l:
                for i in range(self.N_particles):
                    #print(self.time[-1])
                    #print(l.split(':')[1].split('[')[1].split(']')[0].split()[i])
                    self.y[i][len(self.time) - 1] = float(l.split(':')[1].split('[')[1].split(']')[0].split()[i])
            
    def plot_2d(self, y, y_name, y_unit):
        plt.figure()
        x = self.time
        plt.scatter(x, y)
        plt.title('%s as a function of time' % y_name)
        plt.xlabel('Time')
        plt.ylabel('%s (%s)' % (y_name, y_unit))
        plt.grid()
        
    def plot_xy_traj(self):
        plt.figure()
        for i in range(self.N_particles):
            plt.scatter(self.x[i], self.y[i], c=plt.cm.GnBu(np.linspace(0, 1, len(self.time))))
        plt.title('Trajectory of the particles')
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.grid()
        
    def plot_all(self):
        self.plot_2d(self.E_k, 'Kinetic energy', 'J')
        self.plot_2d(self.E_p, 'Potential energy', 'J')
        self.plot_2d(self.E_total, 'Total energy', 'J')
        self.plot_2d(self.temp, 'Temperature', 'K')
        self.plot_2d(self.L, 'Angular momentum', 'X')
        self.plot_xy_traj()
