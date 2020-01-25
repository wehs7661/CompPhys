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
        
        x_approx, p_approx = [x_0], [p_0]

        for i in range(self.n):
            if self.integrator == 'Euler':
                x_approx.append(x_approx[-1] + p_approx[-1] * self.dt)
                # note that at this point x_approx[-1] is the value that was appended now
                p_approx.append(p_approx[-1] - x_approx[-2] * self.dt)
            elif self.integrator == 'symplectic Euler':
                x_approx.append(x_approx[-1] + p_approx[-1] * self.dt)  # same as Euler
                p_approx.append(p_approx[-1] - x_approx[-1] * self.dt)
            elif self.integrator == 'Verlet':
                p_half = p_approx[-1] - x_approx[-1] * (self.dt) / 2
                x_approx.append(x_approx[-1] + p_half * self.dt)
                p_approx.append(p_half - x_approx[-1] * (self.dt) / 2)
            else:
                print('Error: Invalid integrator! Available options are:')
                print('"Euler", "symplectic Euler", and "Verlet".')
                break

        return x_approx, p_approx        
        

    def SHO_plots_compare(self, exact=True, x1=None, p1=None, dt1=None, \
                          integrator1=None, x2, p2, energy=True):
        if exact is True:
            x1, p1, E1 = self.x_exact, self.p_exact, self.E_exact
            title1 = 'Exact solution'
        if exact = False and x1 is not None and p1 is not None:
            E1 = 0.5 * np.power(x1, 2) + 0.5 * np.power(p1, 2)
            title1 = 'Approximation by %s scheme' % integrator
        else:
            print("Error: invalid/insufficient input parameters!")
            sys.exit()

        E2 = 0.5 * np.power(x2, 2) + 0.5 * np.power(p2, 2)

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
        plt.title('Approximation by %s scheme' % self.integrator)
        plt.xlabel('Dimensionless position')
        plt.ylabel('Dimensionless momentum')
        plt.grid()

        if energy is True:
            # Plotting: the total energy as a function of time
            plt.figure()
            _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            plt.suptitle('The total energy as a function of time')

            plt.subplot(1, 2, 1)
            plt.plot(self.t, E1)
            plt.xlabel('Time')
            plt.ylabel('Dimensionless total energy')
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.plot(np.arange(len(E_2)) * self.dt, E_2, '*')
            plt.xlabel('Time')
            plt.ylabel('Dimensionless total energy')
            if max(abs(E_approx)) >= 10000:
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.title('Approximation by %s scheme' % self.integrator)
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