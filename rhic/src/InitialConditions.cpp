#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include "../include/Macros.h"
#include "../include/Hydrodynamics.h"
#include "../include/DynamicalVariables.h"
#include "../include/Precision.h"
#include "../include/Parameters.h"
#include "../include/Trento.h"
// #include "../include/ViscousBjorken.h"
// #include "../include/AnisoBjorken.h"
// #include "../include/IdealGubser.h"
// #include "../include/ViscousGubser.h"
// #include "../include/AnisoGubser.h"
#include "../include/EquationOfState.h"
#include "../include/Projections.h"
#include "../include/AnisoVariables.h"
#include "../include/Viscosities.h"
#include "../include/OpenMP.h"

// cuda: modified on 7/18/21

using namespace std;


inline int linear_column_index(int i, int j, int k, int nx, int ny)
{
	return i  +  nx * (j  +  ny * k);
}


void set_initial_timelike_Tmunu_components(double t, int nx, int ny, int nz, hydro_parameters hydro)
{
	// cuda:	deleted omp pragma, E_CHECK statement
	//			changed u[s].ux to u->ux[s], etc
	//          believe this runs on CPU, so changing to thread loop isn't necessary
	//			keep macro statements involving BOOST_INVARIANT, but comment BOOST_INVARIANT in Macros.h for now

	for(int k = 2; k < nz + 2; k++)
	{
		for(int j = 2; j < ny + 2; j++)
		{
			for(int i = 2; i < nx + 2; i++)
			{
				int s = linear_column_index(i, j, k, nx + 4, ny + 4);

				precision e_s = e[s];

				precision ux = u->ux[s];
				precision uy = u->uy[s];
			#ifndef BOOST_INVARIANT
				precision un = u->un[s];
			#else
				precision un = 0;
			#endif

				precision ut = sqrt(1.  +  ux * ux  +  uy * uy  +  t * t * un * un);

			#ifdef ANISO_HYDRO
			#ifndef BOOST_INVARIANT
				precision utperp = sqrt(1.  +  ux * ux  +  uy * uy);
				precision zt = t * un / utperp;
				precision zn = ut / t / utperp;
			#else
				precision zt = 0;
				precision zn = 1. / t;
			#endif
				precision pl = q->pl[s];
				precision pt = q->pt[s];
			#else
				equation_of_state_new eos(e_s, hydro.conformal_eos_prefactor);
				precision p = eos.equilibrium_pressure();
			#endif

			#ifdef PIMUNU
				precision pitt = q->pitt[s];
				precision pitx = q->pitx[s];
				precision pity = q->pity[s];
			#ifndef BOOST_INVARIANT
				precision pitn = q->pitn[s];
			#else
				precision pitn = 0;
			#endif
			#else
				precision pitt = 0;
				precision pitx = 0;
				precision pity = 0;
				precision pitn = 0;
			#endif

			#ifdef WTZMU
				precision WtTz = q->WtTz[s];
				precision WxTz = q->WxTz[s];
				precision WyTz = q->WyTz[s];
				precision WnTz = q->WnTz[s];
			#else
				precision WtTz = 0;
				precision WxTz = 0;
				precision WyTz = 0;
				precision WnTz = 0;
			#endif

			#ifdef PI
				precision Pi = q->Pi[s];
			#else
				precision Pi = 0;
			#endif

			#ifdef ANISO_HYDRO
				q->ttt[s] = (e_s + pt) * ut * ut  -   pt  +  (pl - pt) * zt * zt  +  2. * WtTz * zt  +  pitt;
				q->ttx[s] = (e_s + pt) * ut * ux  +  WxTz * zt  +  pitx;
				q->tty[s] = (e_s + pt) * ut * uy  +  WyTz * zt  +  pity;
			#ifndef BOOST_INVARIANT
				q->ttn[s] = (e_s + pt) * ut * un  +  (pl - pt) * zt * zn  +  WtTz * zn  +  WnTz * zt  +  pitn;
			#endif
			#else
				q->ttt[s] = (e_s + p + Pi) * ut * ut  -   p  -  Pi  +  pitt;
				q->ttx[s] = (e_s + p + Pi) * ut * ux  +  pitx;
				q->tty[s] = (e_s + p + Pi) * ut * uy  +  pity;
			#ifndef BOOST_INVARIANT
				q->ttn[s] = (e_s + p + Pi) * ut * un  +  pitn;
			#endif
			#endif
			}
		}
	}
}


void set_initial_anisotropy(int nx, int ny, int nz, hydro_parameters hydro)
{
	// cuda: similar to l.33 

	precision plpt_ratio = hydro.plpt_ratio_initial;
	precision conformal_prefactor = hydro.conformal_eos_prefactor;
	precision t = hydro.tau_initial;

	for(int k = 2; k < nz + 2; k++)
	{
		for(int j = 2; j < ny + 2; j++)
		{
			for(int i = 2; i < nx + 2; i++)
			{
				int s = linear_column_index(i, j, k, nx + 4, ny + 4);

				precision e_s = e[s];

				equation_of_state_new eos(e_s, conformal_prefactor);
				precision p = eos.equilibrium_pressure();

				precision pl = 3. * p * plpt_ratio / (2. + plpt_ratio);
				precision pt = 3. * p / (2. + plpt_ratio);

			#ifdef ANISO_HYDRO						// anisotropic hydro initial conditions
				q->pl[s] = pl;
				q->pt[s] = pt;

			#ifdef LATTICE_QCD						// initialize mean field and anisotropic variables
				precision T = eos.T;
				precision b = eos.equilibrium_mean_field();
				precision mass = T * eos.z_quasi();

				precision lambda0 = T;				// initial guess for anisotropic variables
				precision aT0 = 1.;
				precision aL0 = 1.;

				aniso_variables X = find_anisotropic_variables(e_s, pl, pt, b, mass, lambda0, aT0, aL0);

				if(X.did_not_find_solution)
				{
					aniso_regulation[s] = 1;
				}
				else
				{
					aniso_regulation[s] = 0;
				}

				q->b[s] = b;
				lambda[s] = X.lambda;
				aT[s] = X.aT;
				aL[s] = X.aL;
			#endif

			#ifdef PIMUNU
		  		q->pitt[s] = 0;
		  		q->pitx[s] = 0;
		  		q->pity[s] = 0;
		  		q->pixx[s] = 0;
		  		q->pixy[s] = 0;
		  		q->piyy[s] = 0;
		  	#ifndef BOOST_INVARIANT
		  		q->pitn[s] = 0;
		  		q->pixn[s] = 0;
		  		q->piyn[s] = 0;
		  		q->pinn[s] = 0;
		  	#endif
			#endif

		  	#ifdef WTZMU
		  		q->WtTz[s] = 0;
		  		q->WxTz[s] = 0;
		  		q->WyTz[s] = 0;
		  		q->WnTz[s] = 0;
			#endif

			#else 									// viscous hydro initial conditions

		 	#ifdef PIMUNU
		  		precision ux = u->ux[s];
		  		precision uy = u->uy[s];
		  		precision un = 0;

		  	#ifndef BOOST_INVARIANT
		  		un = u->un[s];
		  	#endif
			  
		  		precision ut = sqrt(1.  +  ux * ux  +  uy * uy  +  t * t * un * un);
		  		precision utperp = sqrt(1.  +  ux * ux  +  uy * uy);
		  		precision zt = t * un / utperp;
		  		precision zn = ut / t / utperp;

		  		spatial_projection Delta(ut, ux, uy, un, t * t);

		  		// pi^\munu = (pl - pt)/3 . (\Delta^\munu + 3.z^\mu.z^\nu)
		  		q->pitt[s] = (pl - pt)/3. * (Delta.Dtt  +  3. * zt * zt);
		  		q->pitx[s] = (pl - pt)/3. * Delta.Dtx;
		  		q->pity[s] = (pl - pt)/3. * Delta.Dty;
		  		q->pixx[s] = (pl - pt)/3. * Delta.Dxx;
		  		q->pixy[s] = (pl - pt)/3. * Delta.Dxy;
		  		q->piyy[s] = (pl - pt)/3. * Delta.Dyy;
		  		q->pinn[s] = (pl - pt)/3. * (Delta.Dnn  +  3. * zn * zn);

		  	#ifndef BOOST_INVARIANT
		  		q->pitn[s] = (pl - pt)/3. * (Delta.Dtn  +  3. * zt * zn);
		  		q->pixn[s] = (pl - pt)/3. * Delta.Dxn;
		  		q->piyn[s] = (pl - pt)/3. * Delta.Dyn;
		  	#endif
		  	#endif

		  	#ifdef PI
		  		q->Pi[s] = (pl + 2.*pt)/3.  -  p;
		  	#endif

		  	#endif
			}
		}
	}
}


void read_block_energy_density_from_file(int nx, int ny, int nz, hydro_parameters hydro)
{
	// cuda: changed u[s].ux to u->ux[s] and up[s].ux to up->ux[s], etc


	// load block energy density file to energy density e[s]
	// and set u, up components to 0 (assumes no flow profile)

	// note: e_block.dat in block format (nested matrix is nz x (ny x nx))
	// and contains a header with grid points (nx,ny,nz)

	// e.g. 2d block format:
	// nx
	// ny
	// nz
	// e(1,1)	...	e(nx,1)
	// ...		...	...
	// e(1,ny)	... e(nx,ny)

	// e.g. 3d block format:
	// nx
	// ny
	// nz
	// e(1,1,1)		...		e(nx,1,1)
	// ...			...		...
	// e(1,ny,1)	... 	e(nx,ny,1)
	// e(1,1,2)		...		e(nx,1,2)
	// ...			...		...
	// e(1,ny,2)	... 	e(nx,ny,2)
	// ...
	// e(1,1,nz)	...		e(nx,1,nz)
	// ...			...		...
	// e(1,ny,nz)	... 	e(nx,ny,nz)


	// see how block file is constructed in set_trento_energy_density_and_flow_profile() in Trento.cpp


	FILE * e_block;
  	e_block = fopen("tables/e_block.dat", "r");

  	if(e_block == NULL)
  	{
  		printf("Error: couldn't open e_block.dat file\n");
  	}

  	int nx_block;
  	int ny_block;
  	int nz_block;

  	fscanf(e_block, "%d\n%d\n%d", &nx_block, &ny_block, &nz_block);

	if(nx != nx_block || ny != ny_block || nz != nz_block)
	{
		printf("read_block_energy_density_file error: hydro grid and block file dimensions are inconsistent\n");
		exit(-1);
	}

	// don't use openmp here (don't think it works when reading a file)
	for(int k = 2; k < nz + 2; k++)
	{
		for(int j = 2; j < ny + 2; j++)
		{
			for(int i = 2; i < nx + 2; i++)
			{
				int s = linear_column_index(i, j, k, nx + 4, ny + 4);

				precision e_s;

				fscanf(e_block, "%lf\t", &e_s);

				e[s] = energy_density_cutoff(hydro.energy_min, e_s);

				u->ux[s] = 0.0;		// zero initial velocity
				u->uy[s] = 0.0;
			#ifndef BOOST_INVARIANT
				u->un[s] = 0.0;
			#endif

				up->ux[s] = 0.0;		// also set up = u
				up->uy[s] = 0.0;
			#ifndef BOOST_INVARIANT
				up->un[s] = 0.0;
			#endif
			}
		}
	}
	fclose(e_block);
}



void set_trento_energy_density_profile_from_memory(int nx, int ny, int nz, hydro_parameters hydro, std::vector<double> trento)
{
	// cuda: removed omp pragma
	//		 changed u[s].ux to u->ux[s] and up[s].ux to up->ux[s], etc
	
	precision t0 = hydro.tau_initial;

	if(trento.size() == 0)
	{
		printf("set_trento_energy_density_profile_from_memory error: trento energy density profile is empty (initial condition type only compatible with JETSCAPE)\n");
		exit(-1);
	}

	if((nx * ny * nz) != trento.size())
	{
		printf("set_trento_energy_density_profile_from_memory error: physical grid points and trento energy density vector size are inconsistent\n");
		exit(-1);
	}

  	for(int k = 2; k < nz + 2; k++)
	{
		for(int j = 2; j < ny + 2; j++)
		{
			for(int i = 2; i < nx + 2; i++)
			{
				int s = linear_column_index(i, j, k, nx + 4, ny + 4);
		        int st = linear_column_index(i - 2, j - 2, k - 2, nx, ny);           		// TRENTo vector has no ghost cells

		        e[s] = energy_density_cutoff(hydro.energy_min, trento[st] / (t0 * hbarc));	// convert units to fm^-4, rescale by tau0
		
		        u->ux[s] = 0;		// zero initial velocity
				u->uy[s] = 0;
			#ifndef BOOST_INVARIANT
				u->un[s] = 0;
			#endif

				up->ux[s] = 0;		// also set up = u
				up->uy[s] = 0;
			#ifndef BOOST_INVARIANT
				up->un[s] = 0;
			#endif
			}
		}
	}
}


void set_initial_conditions(precision t, lattice_parameters lattice, initial_condition_parameters initial, hydro_parameters hydro, std::vector<double> trento)
{
	int nx = lattice.lattice_points_x;
	int ny = lattice.lattice_points_y;			// fixed on 6/10/20
	int nz = lattice.lattice_points_eta;

	// cuda: removed dx, dy, dz, dt (believe these were for Gubser test)

	printf("\nInitial conditions = ");

	switch(initial.initial_condition_type)
	{
		// cuda: GPU VAH is only useful for 3+1d simulations, easier to skip configuration for 0+1d or 2+1d simulations
		//		 removed Bjorken and Gubser tests, as well as corresponding files
		//       code quits if choose Bjorken or Gubser initial condition
		//       moved set_initial_anisotropy() and set_initial_timelike_Tmunu_components() after switch statment

		case 1:		// Bjorken
		case 2:		// Gubser
		{
			printf("Bjorken and Gubser tests are currently not available on GPU version\n\n");
			exit(-1);
			break;
		}
		case 3:		// trento (custom version Pb+Pb 2.76 TeV)
		{
			printf("Trento (fluid velocity initialized to zero)\n\n");
			set_trento_energy_density_and_flow_profile(lattice, initial, hydro);
			break;
		}
		case 4:		// read custom energy density block file
		{
			printf("Reading custom energy density block file... (fluid velocity initialized to zero)\n\n");
			read_block_energy_density_from_file(nx, ny, nz, hydro);
			break;
		}
		case 5:		// read trento energy density profile from JETSCAPE C++ vector
		{
			printf("Reading trento energy density profile from JETSCAPE C++ vector... (fluid velocity initialized to zero)\n\n");
			set_trento_energy_density_profile_from_memory(nx, ny, nz, hydro, trento);
			break;
		}
		default:
		{
			printf("\n\nset_initial_conditions error: initial condition type %d is not an option (see initial.properties)\n", initial.initial_condition_type);
			exit(-1);
		}
	}

	set_initial_anisotropy(nx, ny, nz, hydro);
	set_initial_timelike_Tmunu_components(t, nx, ny, nz, hydro);
}


