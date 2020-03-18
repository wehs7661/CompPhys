CompPhys
===================
## Description
This is a repository for the class PHYS-7810: Computational Statistical Physics at CU Boulder. Files in this repository include Lecture notes (in `Lectures` folder), reading materials (in `Resources` folder), in-class tutorials (in `Tutorials` folder), homework (in `Homework` folder), and Python libraries (in `CompPhysLib` folder) used in solving the problems in the tutorials or homework.

## Usage
### Available parameters for MC and MD simulations
#### Basic settings for a simulation
- `simulation`: whether the simulation is MC or MD (Available options: `MC` and `MD`.)
- `N_particles`: the number of the particles in the MC/MD simulation
- `N_steps`: the number of steps of the MC/MD simulation
- `dt`: time step, specifically for MD simulations (ignored if `MC` is chosen)
- `print_freq`: the printing frequency of the results (Default: 1)

#### Conditions and initialization of the simulation
- `dimension`: the number of dimension (Default: 3)
- `box_length`: the size of the cubic simulation box
- `rho`: the reduced number density of the particles (ignored if `box_length` is specified)
- `temperature`: the reducd initial temperature. This must be specified in an MC simulatio but not necessarily for an MD simulatio. It it is not specified in an MD simulation, the velocities will be generted randomly.
- `velo_method`: the method to initialize the velocities of the particles (Available options: `random` and `temp_rescale`. Default: `random`.) The option `temp_rescale` is activated only if `temperature` is specified.
- `coords_method`: the method to initialize the coorinates of the particles (Available options: `random` and `lattice`. Default: `random`.)
- `PBC`: whether the perdioc boundary conditions should be applied (Default: `yes`)
- `max_d`: max displacement, specifically for MC simulations

#### Energy models
- `kb`: Boltzmann constant in reduced form (Default: 1)
- `m`: the reduced mass of the particle given the assumption that all the particles are the same (Default: 1)
- `potential`: the model for the potential energy (Options: `central`, `LJ`)
  - The following parameters are activated only if `central` is chosen:
    - `n`: the order of the total external energy
    - `u`: the coefficient of the total external energy
    - `k`: the order of the pairwise potential 
    - `a`: the coefficient of pariwise central potential
  - The following parameters are activated only if `LJ` is chosen:
    - `epsilon`: the epsilon parameter of Lennard-Jones potential (Default: 1)
    - `sigma`: the sigma paramter of Lennard-Jones potential (Default: 1)

- `energy_trunaction`: whether energy truncation should be applied. (Default: `no`.) If `yes` is specified, this option will activate the following paramters:
  - `r_c`: cutoff distance used in energy truncation (Default: half of the box length)
  - `shift_energy`: whether to shift the truncation (Default: `no`.)

  #### Completed assignments
  As a record, the following tutorials, assignments and exercises are all completed:
  - Tutorial_01, Tutorial_02, Tutorial_03, Tutorial_04. 
  - Homework_01, Homework_02, Homework_03. (All are completed.)
  - Exercise_01.


## Copyright
Copyright (c) 2020, Wei-Tse Hsu (wehs7661@colorado.edu)
