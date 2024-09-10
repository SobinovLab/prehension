#pragma once

#include <math.h>
#include <random>
#include <iostream>
#include "nlopt.h"


class Basinhopping
{
private:
	static void perturb_x_uniform(double* x, int dims, double* lb, double* ub);
	static void perturb_x(double* x, int dims, double* lb, double* ub);

public:
	static double optimize(nlopt_opt& opt, double* x, int dims, double* lb, double* ub,
		void* opt_f, int niter);
};

