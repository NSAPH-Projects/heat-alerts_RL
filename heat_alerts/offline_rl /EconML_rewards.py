
from argparse import ArgumentParser
import numpy as np

from econml.metalearners import XLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# from heat_alerts.Setup_econml import prep_data
from Setup_econml import prep_data

## Set seed?

def main(params):
    ##Setup
    params = vars(params)

    data = prep_data()
    # data = prep_data(eligible=params["eligible"], manual_S_size=params["S_size"])
    X = data[0]
    T = data[1]
    Y = data[2]

    ## Run X-learner:

    # Eventually, adjust model algos?

    est = XLearner(models=GradientBoostingRegressor(),
              propensity_model=GradientBoostingClassifier(),
              cate_models=GradientBoostingRegressor())
    
    est.fit(Y, T, X=X)
    treatment_effects = est.effect(X)
    CATEs = np.stack([treatment_effects, T], axis=1)

    est.fit(Y, T, X=X, inference = "bootstrap")
    treatment_effects = est.effect(X)
    lb, ub = est.effect_interval(X, alpha=0.05) # Bootstrap CIs
    CATEs = np.stack([treatment_effects, lb, ub, T], axis=1)

    np.savetxt("Summer_results/CATEs_6-11.csv", CATEs, delimiter=",")
    np.savetxt("Summer_results/X_small.csv", X, delimiter=",")

    R1_adjust = np.where(T == 1, 0, treatment_effects)
    R0_adjust = np.where(T == 0, 0, -treatment_effects)

    R1_hat = Y + R1_adjust
    R0_hat = Y + R0_adjust

    R_hat = np.stack([R0_hat, R1_hat], axis=1)

    np.savetxt("Summer_results/Modeled_R_6-11.csv", R_hat, delimiter=",")

    # save model / treatment effects?
    # inspect PS?

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="test", help="name to save model under")
    parser.add_argument("--eligible", type=str, default="all", help="days to include in RL")
    parser.add_argument("--S_size", type=str, default="medium", help="Manual size of state matrix")
    parser.add_argument("--regress_model", type=str, default="GB", help="ML algo for regression model")
    parser.add_argument("--PS_model", type=str, default="GB", help="ML algo for propensity score model")
    parser.add_argument("--cate_model", type=str, default="GB", help="ML algo for CATE model")

    args = parser.parse_args()
    main(args)
