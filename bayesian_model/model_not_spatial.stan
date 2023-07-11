functions {
    vector rowSums(matrix R) {
        return R * rep_vector(1, cols(R));
    }
}

data {
    int<lower=0> N;  // number of observations
    int<lower=0> S;  // number of spatial locations
    int<lower=0> DX;  // covariate dimension of time-vary feats
    int<lower=0> DW;  // covariate dimension of space-vary feats
    array[N] int<lower=0> Y;  // response variable
    vector<lower=0>[N] A;  // alerts
    vector<lower=0>[N] offset; // e.g. population
    matrix[N, DX] X;  // time-varying covariates
    matrix[S, DW] W;  // space-varying covariates
    array[N] int<lower=1,upper=S> sind; // spatial location index for each observation
}

parameters {
    matrix<lower=-5, upper=5>[S, DX] beta_unstruct;
    matrix<lower=-5, upper=5>[S, DX] gamma_unstruct;
    // corr_matrix[DX] Sigma_beta;
    // corr_matrix[DX] Sigma_gamma;
    vector<lower=0.01, upper=5>[DX] omega_beta;
    vector<lower=0.01, upper=5>[DX] omega_gamma;
    matrix<lower=-5, upper=5>[DW, DX] delta_beta;
    matrix<lower=-5, upper=5>[DW, DX] delta_gamma;
    // real<lower=0, upper=10> xi;
}

transformed parameters {
    matrix[S, DX] beta = W * delta_beta + beta_unstruct * diag_matrix(omega_beta);
    matrix[S, DX] gamma = W * delta_gamma + gamma_unstruct * diag_matrix(omega_gamma);
    vector[N] lam = exp(rowSums(X .* beta[sind,]));
    vector[N] tau = inv_logit(rowSums(X .* gamma[sind,]));
    vector[N] mu = offset .* lam .* (1.0 - A .* tau);
}

model {
    // Sigma_beta ~ lkj_corr(2);
    // Sigma_gamma ~ lkj_corr(2);

    omega_beta ~ exponential(1);
    omega_gamma ~ exponential(1);

    // for (s in 1:S) {
        // beta_unstruct[s, ] ~ multi_normal(rep_vector(0, DX), Sigma_beta);
        // gamma_unstruct[s, ] ~ multi_normal(rep_vector(0, DX), Sigma_gamma);
    // }
    for (d in 1:DX) {
        beta_unstruct[, d] ~ normal(0, 1.0);
        gamma_unstruct[, d] ~ normal(0, 1.0);
    }
    for (d in 1:DW) {
        delta_beta[, d] ~ normal(0, 1.0);
        delta_gamma[, d] ~ normal(0, 1.0);
    }

    // xi ~ gamma(2, 2)

    Y ~ poisson(mu + 1e-6);
}

