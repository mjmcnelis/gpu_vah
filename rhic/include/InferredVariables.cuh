
#ifndef INFERREDVARIABLES_CUH_
#define INFERREDVARIABLES_CUH_

#include "Precision.h"
#include "DynamicalVariables.h"
#include "Parameters.h"

__global__
void set_inferred_variables_aniso_hydro(const hydro_variables * const __restrict__ q, precision * const __restrict__ e, fluid_velocity * const __restrict__ u, precision t, lattice_parameters lattice, hydro_parameters hydro);

__global__
void set_inferred_variables_viscous_hydro(const hydro_variables * const __restrict__ q, precision * const __restrict__ e, fluid_velocity * const __restrict__ u, precision t, lattice_parameters lattice, hydro_parameters hydro);

#endif