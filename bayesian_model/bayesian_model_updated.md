## Bayesian Model for the Rewards (Medicare Not-Obviously Heat-Related (NOHR) Hospitalizations)
*Goal: model the rewards such that (a) the coefficients are interpretable across space and time, (b) we can explicitly include domain knowledge, (c) uncertainty quantification is straightforward.* 

**Notation**

* Let $s\in\mathcal{S}$ and $t\in\mathcal{T}$ denote a county and time (in days), respectively. Note that time is broken into summers (of 153 days each) and that there are multiple summers per county. 
* Let $Y_{s,t}$ denote the number of NOHR hospitalizations.
* Let $A_{s,t}$ denote whether or not a heat alert was issued.
* Let $X_{s,t}$ denote the temporally-varying covariates at location $s$.
* Let $N_{s,t}$ denote the summer-specific mean of NOHR hospitalizations at location $s$.
* Let $W_{s}$ denote the spatially-varying but temporally static covariates. 
* Let $Y_{s,t}^a$ be the potential outcome for $a\in \{0,1\}$.
* Let $\tau_{s,t}$ be the multiplicative effect of $a=1$.

**General Approach**

* A hierarchical model explaining $\tau_{s,t}$ as a function of the covariates, with the restriction that $\tau_{s,t} \in [0,1]$.
* Local dependence on $(s,t)$ is modulated by heat index and previous heat alerts issued.
* This is the "meta-analysis" approach, and it's meant to solve the spatial confounding problem (such that there is no spatial confounding at the local level). 

**Model**

To make the model scalable we need to assume linearity.

$$
\begin{aligned}
Y_{s,t} & \sim \text{Poisson}(N_{s,t} * \lambda_{s,t} * (1 - A_{s,t}\tau_{s,t})) \\
\lambda_{s,t} & =\beta_{s}^\top X_{s,t} \\
\tau_{s,t} & = \exp(\gamma_{s}^\top X_{s,t} + r_s)
\end{aligned}
$$
