#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/Hydrodynamics.h"
#include "../include/Precision.h"
#include "../include/Parameters.h"

// cuda: modified on 7/18/21 (changed all functions to device, removed inline in sign(), Theta())

__device__
precision sign(precision x)
{
	if(x > 0.)
	{
		return 1.;
	}
	else if(x < 0.)
	{
		return -1.;
	}

	return 0.;
}

__device__
precision Theta(precision x)
{
	if(x > 0.)
	{
		return 1.;
	}

	return 0.;
}

__device__
precision eta_over_s(precision T, hydro_parameters hydro)
{
	precision etas_min = hydro.etas_min;

	if(hydro.temperature_etas == 1)						// temperature dependent piecewise parameterization
	{
		precision aL = hydro.etas_aL * hbarc;			// left slope [fm]
		precision aH = hydro.etas_aH * hbarc;			// right slope [fm]
		precision Tk = hydro.etas_Tk_GeV / hbarc;		// kink temperature fm^-1]
		precision etask = hydro.etas_etask;				// kink etas value

		return fmax(etas_min, etask  +  (T - Tk) * (aL * Theta(Tk - T)  +  aH * Theta(T - Tk)));
	}

	return fmax(etas_min, hydro.constant_etas);
}

__device__
precision zeta_over_s(precision T, hydro_parameters hydro)
{
	precision norm = hydro.zetas_normalization_factor;				// normalization factor
	precision Tpeak = hydro.zetas_peak_temperature_GeV / hbarc;		// peak temperature [fm^-1]
	precision width = hydro.zetas_width_GeV / hbarc;				// width [fm^-1]
	precision skew = hydro.zetas_skew;								// skew

	precision Lambda = width * (1.  +  skew * sign(T - Tpeak));

	return norm * Lambda * Lambda / (Lambda * Lambda  +  (T - Tpeak) * (T - Tpeak));
}