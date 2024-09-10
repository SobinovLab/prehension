#include "Basinhopping.h"

void Basinhopping::perturb_x_uniform(double* x, int dims, double* lb, double* ub)
{
    std::default_random_engine rng;
    for (int i_dim = 0; i_dim < dims; i_dim++) {
        std::uniform_real_distribution<double> distr(lb[i_dim], ub[i_dim]);
        x[i_dim] = distr(rng);
    }
}

void Basinhopping::perturb_x(double* x, const int dims, double* lb, double* ub)
{
    perturb_x_uniform(x, dims, lb, ub);
}

double Basinhopping::optimize(nlopt_opt& opt, double* x, const int dims, double* lb, double* ub, void* opt_f, const int niter)
{
    double minf = HUGE_VAL;
    double iter_minf;
    nlopt_result nlopt_res;

    for (int i_iter = 0; i_iter < niter; i_iter++)
    {
        if ((nlopt_res = nlopt_optimize(opt, x, &iter_minf)) < 0) {
            // iteration failed
            std::cout << "Iteration " << i_iter + 1 << "/" << niter << " failed." << std::endl;
        }
        else {
            // found min
            std::cout << "Iteration " << i_iter + 1 << "/" << niter << " found minimum " << iter_minf << ".";
            if (iter_minf < minf) {
                std::cout << " New minimum!"; 

                minf = iter_minf;

                // perturb the position
                perturb_x(x, dims, lb, ub);
            }
            std::cout << std::endl;
        }
    }

    return minf;
}
