import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np 
import sys
from matplotlib import rc

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': ['DejaVu Sans'],
    'size': 10
})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'default': 'regular'})
plt.rc('font', family='serif')

class SHO_integrators:
    
    def __init__(self):
        # attributes related to the exact solution
        self.t = np.linspace(0, 2*np.pi, 100) 
        self.x_exact = np.sqrt(2) * np.sin(self.t)
        self.p_exact = np.sqrt(2) * np.cos(self.t)
        self.E_exact = 0.5 * np.power(self.x_exact, 2) + 0.5 * np.power(self.p_exact, 2)
        
    def integrators(self, integrator, x_0=0, p_0=np.sqrt(2), n=500, dt=0.01*np.pi*4, reverse=False):
        self.integrator = integrator
        self.n = n
        
        if reverse is False:
            self.dt = dt
        elif reverse is True:
            self.dt = -dt
        
        self.x_approx, self.p_approx = [x_0], [p_0]

        for i in range(self.n):
            if self.integrator == 'Euler':
                self.x_approx.append(self.x_approx[-1] + self.p_approx[-1] * self.dt)
                # note that at this point x_approx[-1] is the value that was appended now
                self.p_approx.append(self.p_approx[-1] - self.x_approx[-2] * self.dt)
            elif self.integrator == 'symplectic Euler':
                self.x_approx.append(self.x_approx[-1] + self.p_approx[-1] * self.dt)  # same as Euler
                self.p_approx.append(self.p_approx[-1] - self.x_approx[-1] * self.dt)
            elif self.integrator == 'Verlet':
                p_half = self.p_approx[-1] - self.x_approx[-1] * (self.dt) / 2
                self.x_approx.append(self.x_approx[-1] + p_half * self.dt)
                self.p_approx.append(p_half - self.x_approx[-1] * (self.dt) / 2)
        
        return self          

    def SHO_plots_compare(self, SHO_obj1=None, SHO_obj2=None, exact=True, energy=True):
        if exact is True:
            x1, p1, E1 = self.x_exact, self.p_exact, self.E_exact
            title1 = 'Exact solution'
            if SHO_obj1 is not None:
                print('Error: SHO_obj1 has been assigned to be the exact solution.')
                print('Specify SHO_obj2 instead.')
                sys.exit()
        if exact is False:
            x1, p1, dt1 = SHO_obj1.x_approx, SHO_obj1.p_approx, SHO_obj1.dt
            E1 = 0.5 * np.power(x1, 2) + 0.5 * np.power(p1, 2)
            title1 = 'Approximation by %s scheme' % SHO_obj1.integrator
            if SHO_obj1 is None and SHO_obj2 is None:
                print('Error: invalid/insufficient input parameters!')
                sys.exit()   

        x2, p2, dt2 = SHO_obj2.x_approx, SHO_obj2.p_approx, SHO_obj2.dt
        E2 = 0.5 * np.power(x2, 2) + 0.5 * np.power(p2, 2)     
        title2 = 'Approximation by %s scheme' % SHO_obj2.integrator

        # Plotting: phase-space trajectory
        plt.figure()
        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        plt.suptitle('Phase-space trajectory of 1D simple harmonic oscillators')

        plt.subplot(1, 2, 1)
        plt.scatter(x1, p1, c = plt.cm.GnBu(np.linspace(0, 1, len(x1))))
        plt.title(title1)
        plt.xlabel('Dimensionless position')
        plt.ylabel('Dimensionless momentum')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.scatter(x2, p2, c=plt.cm.GnBu(np.linspace(0, 1, len(x2))))
        plt.title(title2)
        plt.xlabel('Dimensionless position')
        plt.ylabel('Dimensionless momentum')
        plt.grid()

        # Plotting: energ as a function of time
        if energy is True:
            plt.figure()
            _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            plt.suptitle('The total energy as a function of time')

            plt.subplot(1, 2, 1)
            if exact is True:
                plt.plot(self.t, E1)
            elif exact is False:
                plt.plot(np.arange(len(E1)) * dt1, E1, '.')
            plt.xlabel('Time')
            plt.ylabel('Dimensionless total energy')
            if max(abs(E1)) >= 10000:
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.title(title1)
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.plot(np.arange(len(E2)) * dt2, E2, '.')
            plt.xlabel('Time')
            plt.ylabel('Dimensionless total energy')
            if max(abs(E2)) >= 10000:
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.title(title2)
            plt.grid()
            
    def SHO_plots_reverse(self, x_forward, p_forward, x_backward, p_backward):
        plt.figure()
        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        plt.suptitle('Examination of time-reversibility')

        plt.subplot(1, 2, 1)
        plt.scatter(x_forward, p_forward, c=plt.cm.GnBu(np.linspace(0, 1, len(x_forward))))
        plt.xlabel('Dimensionless position')
        plt.ylabel('Dimensionless momentum')
        plt.title('Time direction: forward')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.scatter(x_backward, p_backward, c=plt.cm.GnBu(np.linspace(0, 1, len(x_backward))))
        plt.xlabel('Dimensionless position')
        plt.ylabel('Dimensionless momentum')
        plt.title('Time direction: backward')
        plt.grid()
