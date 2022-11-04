
library(BART)

## Prep data:

n_days<- 153

data<- read.csv("data/Train_smaller-for-Python.csv")
# load("data/Train-Test.RData")
# data<- Train
# rm("Test")

budget<- data[which(data$dos == 153), "alert_sum"]
Budget<- rep(budget, each = n_days)
data$More_alerts<- Budget - data$alert_sum

DF<- data.frame(scale(data[,vars<- c("HImaxF_PopW", "quant_HI_county", 
                                     "quant_HI_yest_county",
                                     "quant_HI_3d_county", 
                                     "quant_HI_fwd_avg_county",
                                     "Pop_density", "Med.HH.Income",
                                     "year", "dos",
                                     "alert_sum", "More_alerts")]), 
                alert = data$alert,
                dow = data$dow, 
                holiday = data$holiday,
                Zone = data$BA_zone)

dummy_vars<- with(DF, data.frame(model.matrix(~dow+0), model.matrix(~Zone+0)))
DF<- DF[,-which(names(DF) %in% c("dow", "Zone"))]
DF<- data.frame(DF, dummy_vars)

alert_pos<- which(names(DF) == "alert")

X<- DF[,-alert_pos]

## Predict using BART model:
p<- readRDS("Fall_results/BART-model_10-21.rds")

preds<- predict(p, X)

## Only want every other:
# i<- seq(1, 200,2)
# all_probs<- preds$prob.test[i,]
all_probs<- preds$prob.test
means<- colMeans(all_probs)
saveRDS(means, "Fall_results/BART_preds_mean.rds")

near_zero<- means < 0.01
write.csv(near_zero, "Fall_results/BART_preds_near-zero.csv", 
          row.names=FALSE)

saveRDS(all_probs, "Fall_results/BART_preds_all.rds")





