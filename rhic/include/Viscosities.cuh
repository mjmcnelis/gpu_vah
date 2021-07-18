#ifndef VISCOSITIES_H_
#define VISCOSITIES_H_

#include "Precision.h"
#include "Parameters.h"

// cuda: modified on 7/18/21
//       made viscosity functions device

__device__
precision eta_over_s(precision T, hydro_parameters hydro);

__device__
precision zeta_over_s(precision T, hydro_parameters hydro);

#endif