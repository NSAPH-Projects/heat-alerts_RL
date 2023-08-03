
library(arrow)

bayes<- read.csv("Bayesian_models/Bayesian_R_7-19.csv", header=FALSE)

Y<- read_parquet("bayesian_model/data/processed/outcomes.parquet")$other_hosps
A<- read_parquet("bayesian_model/data/processed/actions.parquet")$alert
offset<- read_parquet("bayesian_model/data/processed/offset.parquet")[,1]

# pred_Y<- bayes$R0
# pred_Y[A==1]<- bayes$R1[A==1]
# pred_Y<- pred_Y*offset

pred_Y<- bayes$V3#/offset
Y<- Y/offset

cor(Y, pred_Y)^2
1 - (sum((Y - pred_Y)^2)/(((length(A)-1)*var(Y))))

