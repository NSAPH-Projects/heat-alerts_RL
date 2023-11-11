# Brainstorm of Bayesian Model

**Notation**

* Let $s\in\mathcal{S}$ and $t\in\mathcal{T}$ denote a county and time, respectively. For now, let's ignore that time is broken down in summers and that there are multiple summers per county. Time variables like "days since start of summer" will be variables for the purpose of the model.
* Let $Y_{s,t}$ denote the number of hospitalizations.
* Let $A_{s,t}$ whether or not a heat alert was issued.
* Let $X_ {s,t}$ denote the temporally-varying covariates of location $s$.
* Let $W_{s}$ denote the spatially-varying covaraites. 
* Let $Y_{s,t}^a$ be the potential outcome for $a\in \{0,1\}$.

**Design goals**

* No spatial confounding at the local level. We will learn a conditional effect $\tau_{s,t}$ with the restriction that $\tau_{s,t} \geq 0$.
* Local dependence on $(s,t)$ modulated by previous alerts, previous hospitalizations, heat index, air pollution, weather patterns.
* A hiercachical model explaining $\tau_{s,t}$ as a function of the covariates. 
* This is the "meta-analysis" approach, and it's meant to solve the spatial confounding approach.
* The model should account for potential zero-inflation.


**Model**

To make the model scalable we need to assume linearity.

$$
\begin{aligned}
Y_{s,t} & \sim \text{NegBin}(N_{st} (\lambda_{s,t} + A_{s,t}\tau_{s,t}), \rho) \\
\lambda_{s,t} & =\beta_{s}^\top X_{s,t} \\
\tau_{s,t} & = \exp(\gamma_{s}^\top X_{s,t} + r_s)
\end{aligned}
$$

where $\rho$ is an overdispersion parameter, $N_{s,t}$ is the relevant population of that given day (fixed in time for practical purposes). Here $\tau_{s,t}$ is the scaled "effect" of the heat alert, and is the one we are interested in. This effect is explained 
by the time-varying covariates and modulated with a random effect $r_s$.

Here we encode our knowledge that alerts are not harmful by using the exponential link function for $\tau_{s,t}$.

To considered non-linear effects we can use spline basis for $X_{s,t}$. We can of course consider an intercept term, but not written here to keep the notation simple (e.g., $X_{s,t}$ includes a column of ones).

## Choice of priors

### Basic hierarchical priors

We can consider a hierarchical model, where the parameters are drawn from a common distribution. For example:

$$
\begin{aligned}
\beta_{s} & \sim \text{Normal}(\mu_\beta, \Sigma_\beta) \\
r_{s} & \sim \text{Normal}(\mu_r, \sigma_r^2) \\
\rho & \sim \text{Gamma}(a_\rho, b_\rho)
\end{aligned}
$$

The Gamma prior for $\rho$ is a common choice for overdispersion parameters. A dispersion parameter of $\rho=1$ corresponds to a Poisson distribution; smaller values of $\rho$ correspond to more overdispersion.

### Accounting for spatial correlations

The question arise, do we need some form of smoothing on the parameters, and potentially account for spatial correlations?

One possibility is to use the Besag-York-Mollie (BYM) model, which is a popular model for spatial data. The BYM model is a hierarchical model that decomposes the spatial variation into two components: a spatially structured component and a spatially unstructured component. The spatially structured component is modeled as a Gaussian Markov random field (GMRF) with a conditional autoregressive (CAR) prior. The spatially unstructured component is modeled as an independent Gaussian random field. The BYM model is written as:

$$
\begin{aligned}
\beta_{s} &= \beta_{s}^\text{spatial} + \beta_{s}^\text{unstruct} \\
r_{s} &= r_{s}^\text{spatial} + r_{s}^\text{unstruct} \\
\beta_{s}^\text{spatial} &\sim \text{CAR}(\xi_\beta) \\
r_{s}^\text{spatial} &\sim \text{CAR}(\xi_r) \\
\beta_{s}^\text{unstruct} &\sim \text{Normal}(\mu_\beta, \Sigma_\beta) \\
r_{s}^\text{unstruct} &\sim \text{Normal}(\mu_r, \sigma_r^2) \\
\end{aligned}
$$

The relative scaling of the spatial and unstructured parameters are important, and thoroughly discussed in the BYM literature.



### Account for zero-inflation

If the model above fails to account for zero-inflation, we can consider a zero-inflated negative binomial model. The zero-inflated negative binomial model is written as:

$$
\begin{aligned}
Y_{s,t} & \sim \pi_s \delta_0 + (1 - \pi_s)\text{NegBin}(N_{st} (\lambda_{s,t} + A_{s,t}\tau_{s,t}), \rho) \\
\lambda_{s,t} & =\beta_{s}^\top X_{s,t} \\
\tau_{s,t} & = \exp(\gamma_{s}^\top X_{s,t} + r_s) \\
\text{logit}(\pi_{s,t}) & = \alpha_{s}^\top W_{s,t} \\
\end{aligned}
$$

where $\pi_{s}$ is the probability of a zero-inflation in a given location. We can set a prior $\pi_{s} \sim \text{Beta}(a_\pi, b_\pi)$ that encourages some sparsity.

### Covariate-driven priors / Meta-analysis
