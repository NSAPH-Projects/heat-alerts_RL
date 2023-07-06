functions {
    real icar_normal_lpdf(vector phi, int S, array[] int node1, array[] int node2) {
        return -0.5 * dot_self(phi[node1] - phi[node2]) + normal_lpdf(sum(phi) | 0, 0.001 * S);
    }
    vector rowSums(matrix R) {
        int D = cols(R);
        return R * rep_vector(1, D);
    }
}

data {
    int<lower=0> N;  // number of observations
    int<lower=0> S;  // number of spatial locations
    int<lower=0> DX;  // covariate dimension of time-vary feats
    int<lower=0> DW;  // covariate dimension of space-vary feats
    int<lower=0> M;  // number of edges
    array[M] int<lower=1,upper=S> node1;  // node1
    array[M] int<lower=1,upper=S> node2;  // node2
    array[N] int<lower=0> y;  // response variable
    vector<lower=0>[N] A;  // alerts
    vector<lower=0>[S] P; // relevant population
    matrix[N, DX] X;  // time-varying covariates
    matrix[S, DW] W;  // space-varying covariates
    array[N] int<lower=1,upper=S> sind; // spatial location index for each observation
}

parameters {
    matrix<lower=-5, upper=5>[S, DX] beta_spatial;
    matrix<lower=-5, upper=5>[S, DX] beta_unstruct;
    matrix<lower=-5, upper=5>[S, DX] gamma_spatial;
    matrix<lower=-5, upper=5>[S, DX] gamma_unstruct;
    corr_matrix[DX] Sigma_beta;
    corr_matrix[DX] Sigma_gamma;
    vector<lower=0, upper=10>[DX] omega_beta;
    vector<lower=0, upper=10>[DX] omega_gamma;
    matrix<lower=-5, upper=5>[DW, DX] delta_beta;
    matrix<lower=-5, upper=5>[DW, DX] delta_gamma;
    real<lower=0, upper=1> rho_beta;
    real<lower=0, upper=1> rho_gamma;
    real<lower=0, upper=10> xi;
}

transformed parameters {
    matrix[S, DX] beta;
    matrix[S, DX] gamma;
    for (s in 1:S) {
        beta[s,] = W[s,] * delta_beta + to_row_vector(omega_beta) .* (rho_beta * (beta_spatial[s,]) + sqrt(1 - rho_beta) * (beta_unstruct[s,]));
        gamma[s,] = W[s,] * delta_gamma + to_row_vector(omega_gamma) .* (rho_gamma * (gamma_spatial[s,]) + sqrt(1 - rho_gamma) * (gamma_unstruct[s,]));
    }

    vector[N] lam = exp(rowSums(X .* beta[sind,]));
    vector[N] tau = inv_logit(rowSums(X .* gamma[sind,]));
    vector[N] mu = P[sind] .* lam .* (1.0 - A .* tau);
}

model {
    Sigma_beta ~ lkj_corr(2);
    Sigma_gamma ~ lkj_corr(2);

    omega_beta ~ cauchy(0, 2.5);
    omega_gamma ~ cauchy(0, 2.5);

    for (d in 1:DX) {
        target += icar_normal_lpdf(beta_spatial[, d] | S, node1, node2);
    }
    for (d in 1:DW) {
        target += icar_normal_lpdf(gamma_spatial[, d] | S, node1, node2);
    }
    for (s in 1:S) {
        beta_unstruct[s, ] ~ multi_normal(rep_vector(0, DX), Sigma_beta);
        gamma_unstruct[s, ] ~ multi_normal(rep_vector(0, DX), Sigma_gamma);
    }
    for (d in 1:DW) {
        delta_beta[, d] ~ normal(0, 1);
        delta_gamma[, d] ~ normal(0, 1);
    }

    rho_beta ~ beta(2, 2);
    rho_gamma ~ beta(2, 2);
    xi ~ gamma(2, 2);

    y ~ neg_binomial_2(mu + 1e-6, xi);
}

