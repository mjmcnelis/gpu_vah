
#ifndef REGULATION_CUH_
#define REGULATION_CUH_

#include "Precision.h"
#include "DynamicalVariables.h"
#include "Parameters.h"

__global__
void regulate_residual_currents(precision t, hydro_variables * const __restrict__ q, precision * const __restrict__ e, const fluid_velocity * const __restrict__ u, lattice_parameters lattice, hydro_parameters hydro, int RK2);

__global__
void regulate_viscous_currents(precision t, hydro_variables * const __restrict__ q, precision * const __restrict__ e, const fluid_velocity * const __restrict__ u, lattice_parameters lattice, hydro_parameters hydro, int RK2);

#endif


