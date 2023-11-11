# Brainstorm of Bayesian Model

**Notation**

* Let $s\in\mathcal{S}$ and $t\in\mathcal{T}$ denote a county and time, respectively. For now, let's ignore that time is broken down in summers and that there are multiple summers per county. Time variables like "days since start of summer" will be variables for the purpose of the model.
* Let $Y_{s,t}$ denote the number of hospitalizations.
* Let $A_{s,t}$ whether or not a heat alert was issued.
* Let $X_ {s,t}$ denote the time-varying covariates of location $s$.
* Let $W_{s}$ denote the space-varying covariates. 
* Let $Y_{s,t}^a$ be the potential outcome for $a\in \{0,1\}$.

**Note**. To avoid writing intercepts in the model, assume $X_{s,t}$ and $W_s$ have a column of ones.

<!-- **Design goals**

* No spatial confounding at the local level. We will learn a conditional effect $\tau_{s,t}$ with the restriction that $\tau_{s,t} \geq 0$.
* Local dependence on $(s,t)$ modulated by previous alerts, previous hospitalizations, heat index, air pollution, weather patterns.
* A hiercachical model explaining $\tau_{s,t}$ as a function of the covariates. 
* This is the "meta-analysis" approach, and it's meant to solve the spatial confounding approach.
* The model should account for potential zero-inflation. -->


**Model**


$$
\begin{aligned}
Y_{s,t} & \sim \text{NegBin}(N_{st}\mu_{s,t}, \rho) \\
\mu_{s,t} & = \lambda_{s,t}(1 - A_{s,t}\tau_{s,t}) \\
\lambda_{s,t} & = \exp(\beta_{s}^\top X_{s,t}) \\
\tau_{s,t} & = \text{sigmoid}(\gamma_{s}^\top X_{s,t}) \\
\rho & \sim \text{Gamma}(a_\rho, b_\rho) \\
\end{aligned} 
$$

where $\rho$ is an overdispersion parameter, $N_{s,t}$ is the relevant population of that given day (fixed in time for practical purposes). Here $\tau_{s,t}$ is the effect of the heat alert. This effect is explained 
by the time-varying covariates and modulated with a random effect $r_s$.

Here we encode our knowledge that alerts are not harmful by using the exponential link function for $\tau_{s,t}$.

**Note about Intercepts**. To avoid writing intercepts in the model, assume $X_{s,t}$ and $W_s$ have a column of ones.

## Choice of priors

### Basic hierarchical priors

We can consider a hierarchical model, where the parameters are drawn from a common distribution. For example:

$$
\begin{aligned}
\beta_{s} & \sim \text{Normal}(\mu_\beta, \Sigma_\beta) \\
\gamma_{s} & \sim \text{Normal}(\mu_\gamma, \Sigma_\gamma^2) \\
\mu_\beta & \sim 1 \\
\mu_\gamma & \sim 1 \\
\Sigma_\beta & \sim \text{quadform}(\text{LKJ}(2), \omega_\beta) \\
\Sigma_\gamma & \sim \text{quadform}(\text{LKJ}(2), \omega_\gamma) \\
\omega_\beta & \sim \text{HalfCauchy}(5) \\
\omega_\gamma & \sim \text{HalfCauchy}(5) \\
% \omega_\beta, \omega_\gamma &\sim \text{HalfCauchy}(5) \\
% \rho_\beta,  \rho_\gamma &\sim \text{Beta}(a_\rho, b_\rho)  \\
\end{aligned}
$$
where LKJ is a prior over correlation matrices. LKJ is commonly used in Stan models, recommended by Andrew Gelman. 

### Adding spatial structure

Do we need some form of smoothing?

One possibility is to use the Besag-York-Mollie (BYM) model (for smoothing, not spatial errors!), which is a popular model for spatial data. The BYM model is a hierarchical model that decomposes the spatial variation into two components: a spatially structured component and a spatially unstructured component. The spatially structured component is modeled as a Gaussian Markov random field (GMRF) with a conditional autoregressive (CAR) prior. The spatially unstructured component is modeled as an independent Gaussian random field. The BYM model is written as:

$$
\begin{aligned}
\beta_{s} &= \omega_\beta (\rho_\beta \beta_{s}^\text{spatial} + \sqrt{1 - \rho}_\beta\beta_{s}^\text{unstruct}) \\

\gamma_{s} &= \omega_\gamma(\rho_\gamma \gamma_{s}^\text{spatial} + \sqrt{1 - \rho_\gamma} \gamma_{s}^\text{unstruct}) \\
\beta_{s}^\text{spatial} &\sim \text{CAR}(\Omega) \\
\gamma_{s}^\text{spatial} &\sim \text{CAR}(\Omega) \\
\beta_{s}^\text{unstruct} &\sim \text{Normal}(0, \Sigma_\beta^2) \\
\gamma_{s}^\text{unstruct} &\sim \text{Normal}(0, \Sigma_\gamma^2) \\
\Sigma_\beta & \sim \text{LKJ}(2) \\
\Sigma_\gamma & \sim \text{LKJ}(2) \\
\omega_\beta & \sim \text{HalfCauchy}(5) \\
\omega_\gamma & \sim \text{HalfCauchy}(5) \\
\end{aligned}
$$
  
$\Omega$ is the scaled graph Laplacian of the adjacency matrix, obtained by dividing by the average number of neighbors, so that the marginal variance of $\beta_s^\text{spatial}$ and $\gamma_s^\text{spatial}$ is 1.


### Account for zero-inflation

If the model above fails to account for zero-inflation, we can consider a zero-inflated negative binomial model. The zero-inflated negative binomial model is written as:

$$
\begin{aligned}
Y_{s,t} & \sim \pi_s \delta_0 + (1 - \pi_s)\text{NegBin}(N_{st}\mu_{s,t}, \rho)  \\
\pi_s &\sim \text{Beta}(a_\pi, b_\pi) \\
\end{aligned}
$$




### Covariate-driven priors / Meta-analysis

We can decrease the uncertainty in the random effects by using a meta-analysis approach. 

$$
\begin{aligned}
\beta_s &= \delta_\beta' W_s + \omega_s(\rho_s \beta_{s}^\text{spatial} + \sqrt{1 - \rho}_s\beta_{s}^\text{unstruct}) \\
\gamma_s &= \delta_\gamma' W_s + \omega_s(\rho_s \gamma_{s}^\text{spatial} + \sqrt{1 - \rho}_s \gamma_{s}^\text{unstruct}) \\

\end{aligned}
$$

where $\delta_\beta, \delta_\gamma$ are vectors of regression coefficients. The idea is that the spatial covariates will explain some of the spatial variation, and the random effects will explain the rest. 
