
#ifndef FLUXTERMS_CUH_
#define FLUXTERMS_CUH_

#include "Precision.h"

// cuda: modified on 7/18/21
//       made flux functions device

// precision approximate_derivative(precision qm, precision q, precision qp);

__device__
precision compute_max_local_propagation_speed(const precision * const __restrict__ v_data, precision v, precision Theta);

__device__
void flux_terms(precision * const __restrict__ Hp, precision * const __restrict__ Hm, const precision * const __restrict__ q_data, const precision * const __restrict__ q1_data, const precision * const __restrict__ q2_data, const precision * const __restrict__ v_data, precision v, precision Theta);


#endif


