
library(BART)

## Prep data:

n_days<- 153

load("data/Small_S-A-R_prepped.RData")

## Predict using BART model:
p<- readRDS("Fall_results/BART-model_11-20.rds")

X<- DF

X$holiday1<- 1
X$holiday1[X$holiday == 1]<- 0
X$holiday2<- 0
X$holiday2[X$holiday == 1]<- 1

preds<- predict(p, X[,names(p$treedraws$cutpoints)])

## Only want every other:
# i<- seq(1, 200,2)
# all_probs<- preds$prob.test[i,]
all_probs<- preds$prob.test
means<- colMeans(all_probs)
saveRDS(means, "Fall_results/BART_preds_mean_11-20.rds")

near_zero<- means < 0.01
write.csv(near_zero, "Fall_results/BART_preds_near-zero_11-20.csv", 
          row.names=FALSE)

saveRDS(all_probs, "Fall_results/BART_preds_all_11-20.rds")





