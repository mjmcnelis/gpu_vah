#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/Macros.h"
#include "../include/DynamicalVariables.h"
#include "../include/InferredVariables.h"
#include "../include/Projections.h"
#include "../include/OpenMP.h"

// CUDA: got rid of sqrt_two, sqrt_three, inline for these two functions
__device__ 
int linear_column_index(int i, int j, int k, int nx, int ny)
{
	return i  +  nx * (j  +  ny * k);
}

__device__ 
precision pressure_cutoff(precision p_min, precision p)
{
	precision p_cut = fmax(0., p);

	return p_cut  +  p_min * exp(- p_cut / p_min);	// regulate pressure, asymptotes to p_min as pl,pt -> 0
}

__global__
void regulate_residual_currents(precision t, hydro_variables * const __restrict__ q, precision * const __restrict__ e, const fluid_velocity * const __restrict__ u, lattice_parameters lattice, hydro_parameters hydro, int RK2)
{
#ifdef ANISO_HYDRO
	int nx = lattice.lattice_points_x;
	int ny = lattice.lattice_points_y;

	precision pressure_min = hydro.pressure_min;

	precision t2 = t * t;
	precision t4 = t2 * t2;

	int pl_reg = 0;
	int pt_reg = 0;
	int pt_reg_conformal = 0;
	int b_reg = 0;

	// CUDA: need to review this
	unsigned int threadID = threadIdx.x  +  blockDim.x * blockIdx.x;

	// CUDA: do I need to define d_nElements?
	if(threadID < d_nElements)
	{
		// CUDA: what does this mean exactly? do I really need d_ncx, d_ncy? is unsigned int necessary?
		unsigned int k = 2  +  (threadID / (d_nx * d_ny));
		unsigned int j = 2  +  (threadID % (d_nx * d_ny)) / d_nx;
		unsigned int i = 2  +  (threadID % d_nx);
 
		unsigned int s = linear_column_index(i, j, k, nx + 4, ny + 4);

		// CUDA: can I use q.pl[s] or no?
		precision e_s = e[s];
		precision pl  = q->pl[s];
		precision pt  = q->pt[s];

		precision pavg = (pl + 2.*pt)/3.;

	#ifdef MONITOR_PLPT
		plpt_regulation[s] = 0;					// 0 = no regulation

		if(pl < pressure_min)
		{
			pl_reg++;

			plpt_regulation[s] += 1;			// 1 = pl regulated
		}
		if(pt < pressure_min)
		{
			pt_reg++;

			plpt_regulation[s] += 2;			// 2 = pt regulated (3 = both regulated)
		}
	#endif

		pl = pressure_cutoff(pressure_min, pl);	// pl, pt regulation
		pt = pressure_cutoff(pressure_min, pt);

	#ifdef CONFORMAL_EOS
		pt = (e_s - pl) / 2.;

		if(pt < pressure_min)
		{
			pt_reg_conformal++;
			pt = pressure_cutoff(pressure_min, pt);  // is this still a problem?
		}
	#endif

		q->pl[s] = pl;							// regulate longitudinal and transverse pressures
		q->pt[s] = pt;

	#ifdef B_FIELD 								// regulate non-equilibrium mean field component
		precision b = q->b[s];
		equation_of_state_new eos(e_s, hydro.conformal_eos_prefactor);
		precision beq = eos.equilibrium_mean_field();

		precision db = b - beq;

	#ifdef MONITOR_B
		b_regulation[s] = 0;
	#endif

		if(db < 0. && fabs(beq / db) < 1.)
		{
			db *= fabs(beq / db);
			b_reg += 1;

		#ifdef MONITOR_B
			b_regulation[s] = 1;
		#endif
		}

		q->b[s] = beq + db;
	#endif

	#if (NUMBER_OF_RESIDUAL_CURRENTS != 0)		// regulate residual currents
		precision ux = u->ux[s];
		precision uy = u->uy[s];
	#ifndef BOOST_INVARIANT
		precision un = u->un[s];
	#else
		precision un = 0;
	#endif

		precision ut = sqrt(1.  +  ux * ux  +  uy * uy  +  t2 * un * un);
		precision utperp = sqrt(1.  +  ux * ux  +  uy * uy);

	#ifndef BOOST_INVARIANT
		precision zt = t * un / utperp;
		precision zn = ut / t / utperp;
		precision Taniso = sqrt(pl * pl  +  2. * pt * pt);
	#else
		precision zt = 0;
		precision zn = 1. / t;
		precision Taniso = sqrt(2.) * pt;
	#endif

	#ifdef PIMUNU
		precision pitt = q->pitt[s];
		precision pitx = q->pitx[s];
		precision pity = q->pity[s];
		precision pixx = q->pixx[s];
		precision pixy = q->pixy[s];
		precision piyy = q->piyy[s];

	#ifndef BOOST_INVARIANT
		precision pitn = q->pitn[s];
		precision pixn = q->pixn[s];
		precision piyn = q->piyn[s];
		precision pinn = q->pinn[s];
	#else
		precision pitn = 0;
		precision pixn = 0;
		precision piyn = 0;
		precision pinn = 0;
	#endif
		// enforce orthogonality and tracelessness
		piyy = (- pixx * (1.  +  uy * uy)  +  2. * pixy * ux * uy) / (1.  +  ux * ux);
		pitx = (pixx * ux  +  pixy * uy) * ut / (utperp * utperp);
		pity = (pixy * ux  +  piyy * uy) * ut / (utperp * utperp);
		pixn = pitx * un / ut;
		piyn = pity * un / ut;
		pitn = (pixn * ux  +  piyn * uy) * ut / (utperp * utperp);
		pinn = pitn * un / ut;
		pitt = (pitx * ux  +  pity * uy  +  t2 * pitn * un) / ut;
	#else
		precision pitt = 0;
		precision pitx = 0;
		precision pity = 0;
		precision pitn = 0;
		precision pixx = 0;
		precision pixy = 0;
		precision pixn = 0;
		precision piyy = 0;
		precision piyn = 0;
		precision pinn = 0;
	#endif

	#ifdef WTZMU
		precision WtTz = q->WtTz[s];
		precision WxTz = q->WxTz[s];
		precision WyTz = q->WyTz[s];
		precision WnTz = q->WnTz[s];

		// enforce orthogonality
		WtTz = (WxTz * ux  +  WyTz * uy) * ut / (utperp * utperp);
		WnTz = WtTz * un / ut;
	#else
		precision WtTz = 0;
		precision WxTz = 0;
		precision WyTz = 0;
		precision WnTz = 0;
	#endif

		precision factor_pi = 1;
		precision factor_W = 1;

	#ifdef MONITOR_REGULATIONS
		viscous_regulation[s] = 0;
	#endif

		precision Tres = sqrt(fabs(-2. * (WtTz * WtTz  -  WxTz * WxTz  -  WyTz * WyTz  -  t2 * WnTz * WnTz)  +  pitt * pitt  +  pixx * pixx  +  piyy * piyy  +  t4 * pinn * pinn  -  2. * (pitx * pitx  +  pity * pity  -  pixy * pixy  +  t2 * (pitn * pitn  -  pixn * pixn  -  piyn * piyn))));

		precision factor = fabs(Taniso / (1.e-8 + Tres));

		if(factor < 1.)
		{
			factor_pi = factor;
			factor_W = factor;
			
		#ifdef MONITOR_REGULATIONS
			viscous_regulation[s] = 1;
		#endif
		}

	#ifdef PIMUNU
		q->pitt[s] = factor_pi * pitt;
		q->pitx[s] = factor_pi * pitx;
		q->pity[s] = factor_pi * pity;
		q->pixx[s] = factor_pi * pixx;
		q->pixy[s] = factor_pi * pixy;
		q->piyy[s] = factor_pi * piyy;

	#ifndef BOOST_INVARIANT
		q->pitn[s] = factor_pi * pitn;
		q->pixn[s] = factor_pi * pixn;
		q->piyn[s] = factor_pi * piyn;
		q->pinn[s] = factor_pi * pinn;
	#endif
	#endif

	#ifdef WTZMU
		q->WtTz[s] = factor_W * WtTz;
		q->WxTz[s] = factor_W * WxTz;
		q->WyTz[s] = factor_W * WyTz;
		q->WnTz[s] = factor_W * WnTz;
	#endif

	#endif
	}
#endif
}


__global__
void regulate_viscous_currents(precision t, hydro_variables * const __restrict__ q, precision * const __restrict__ e, const fluid_velocity * const __restrict__ u, lattice_parameters lattice, hydro_parameters hydro, int RK2)
{
#ifndef ANISO_HYDRO
#if (NUMBER_OF_VISCOUS_CURRENTS != 0)

	// CUDA: is passing these parameters okay? do I need to cuda configure d_ncx, d_nx, etc directly?
	int nx = lattice.lattice_points_x;
	int ny = lattice.lattice_points_y;
	// int nz = lattice.lattice_points_eta;

	precision t2 = t * t;
	precision t4 = t2 * t2;

	// CUDA: need to review this
	unsigned int threadID = threadIdx.x  +  blockDim.x * blockIdx.x;

	// CUDA: do I need to define d_nElements? (d_ stands for device)
	if(threadID < d_nElements)
	{
		// CUDA: what does this mean exactly? do I really need d_ncx, d_ncy?
		// CUDA: is the linear column indexing the same as GPU-VH? or did I change this, too?
		unsigned int k = 2  +  (threadID / (d_nx * d_ny));
		unsigned int j = 2  +  (threadID % (d_nx * d_ny)) / d_nx;
		unsigned int i = 2  +  (threadID % d_nx);

		// CUDA: do I need d_nx, d_ny?
		unsigned int s = linear_column_index(i, j, k, nx + 4, ny + 4);

		// CUDA: should eos class/functions be device or device+host?
		equation_of_state_new eos(e[s], hydro.conformal_eos_prefactor);		
		precision p = eos.equilibrium_pressure();

		// CUDA data structure of fluid and hydro structs need to change to u->ux[s], etc
		precision ux = u->ux[s];
		precision uy = u->uy[s];

	#ifndef BOOST_INVARIANT
		precision un = u->un[s];
	#else
		precision un = 0;
	#endif
		precision ut = sqrt(1.  +  ux * ux  +  uy * uy  +  t2 * un * un);
		precision utperp = sqrt(1.  +  ux * ux  +  uy * uy);

		precision Teq = sqrt(3.) * p;

	#ifdef PIMUNU
		precision pitt = q->pitt[s];
		precision pitx = q->pitx[s];
		precision pity = q->pity[s];
		precision pixx = q->pixx[s];
		precision pixy = q->pixy[s];
		precision piyy = q->piyy[s];
		precision pinn = q->pinn[s];
	#ifndef BOOST_INVARIANT
		precision pitn = q->pitn[s];
		precision pixn = q->pixn[s];
		precision piyn = q->piyn[s];
	#else
		precision pitn = 0;
		precision pixn = 0;
		precision piyn = 0;
	#endif
		// enforce orthogonality and tracelessness
		pinn = (pixx * (ux * ux  -  ut * ut)  +  piyy * (uy * uy  -  ut * ut)  +  2. * (pixy * ux * uy  +  t2 * un * (pixn * ux  +  piyn * uy))) / (t2 * utperp * utperp);
		pitn = (pixn * ux  +  piyn * uy  +  t2 * un * pinn) / ut;
		pity = (pixy * ux  +  piyy * uy  +  t2 * un * piyn) / ut;
		pitx = (pixx * ux  +  pixy * uy  +  t2 * un * pixn) / ut;
		pitt = (pitx * ux  +  pity * uy  +  t2 * un * pitn) / ut;
	#else
		precision pitt = 0;
		precision pitx = 0;
		precision pity = 0;
		precision pitn = 0;
		precision pixx = 0;
		precision pixy = 0;
		precision pixn = 0;
		precision piyy = 0;
		precision piyn = 0;
		precision pinn = 0;
	#endif
	#ifdef PI
		precision Pi = q->Pi[s];
	#else
		precision Pi = 0;
	#endif
		precision factor_pi = 1;
		precision factor_bulk = 1;

	#ifdef MONITOR_REGULATIONS
		if(!RK2)
		{	// most regulations occur after first intermediate Euler step (very few after RK2 averaging)
			viscous_regulation[s] = 0;
		}
	#endif
		// CUDA: got rid of switch (only using one regulation method)
		precision Tvisc = sqrt(fabs(3. * Pi * Pi  + pitt * pitt  +  pixx * pixx  +  piyy * piyy  +  t4 * pinn * pinn  -  2. * (pitx * pitx  +  pity * pity  -  pixy * pixy  +  t2 * (pitn * pitn  -  pixn * pixn  -  piyn * piyn))));

		precision factor = fabs(Teq / (1.e-10 + Tvisc));

		if(factor < 1.)
		{
			factor_pi = factor;
			factor_bulk = factor;

		#ifdef MONITOR_REGULATIONS
			if(!RK2)
			{
				viscous_regulation[s] = 1;
			}
		#endif
		}

		#ifdef PIMUNU
			q->pitt[s] = factor_pi * pitt;
			q->pitx[s] = factor_pi * pitx;
			q->pity[s] = factor_pi * pity;
			q->pixx[s] = factor_pi * pixx;
			q->pixy[s] = factor_pi * pixy;
			q->piyy[s] = factor_pi * piyy;
			q->pinn[s] = factor_pi * pinn;
		#ifndef BOOST_INVARIANT
			q->pitn[s] = factor_pi * pitn;
			q->pixn[s] = factor_pi * pixn;
			q->piyn[s] = factor_pi * piyn;
		#endif
		#endif

		#ifdef PI
			q->Pi[s] = factor_bulk * Pi;
		#endif
	}
#endif
#endif
}