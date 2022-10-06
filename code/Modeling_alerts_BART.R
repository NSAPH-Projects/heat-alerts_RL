
library(BART)
library(profvis)
library(ggplot2)
library(parallel)
detectCores()

## Read in data and make dummy vars:

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

## Set up train and "test" sets:

set.seed(321)
train_inds<- sample(1:nrow(data), 0.1*nrow(data), replace = FALSE)

alert_pos<- which(names(DF) == "alert")
x.train<- DF[train_inds, -alert_pos]
y.train<- DF[train_inds, "alert"]

test_inds<- setdiff(1:nrow(data), train_inds)

x.test<- DF[test_inds, -alert_pos]
y.test<- DF[test_inds, "alert"]


## Set up BART:

M<- 1000 # number of posterior samples (1000 is the default, after 100 burn-in)
B<- 40 # number of cores (on the cluster)

start<- Sys.time()
post<- pbart(x.train, y.train, x.test, printevery = 10)
# post<- mc.pbart(x.train, y.train, x.test, ndpost = M, nskip = 6000,
#                 mc.cores = B, seed = 321, 
#                 printevery = 10)
# saveRDS(post, "Fall_results/BART-post_10-6.rds")
end<- Sys.time()
end - start
## If we use the argument "sparse=TRUE", then can look at post$varprob...

## Look at the posterior samples: post$yhat.train and post$yhat.test
  ## If using pbart, need to transform the posterior samples with pnorm() to get probs

preds.train<- pnorm(post$yhat.train)
preds.test<- pnorm(post$yhat.test)


#### Check convergence:

plot(1:nrow(preds.train), preds.train[,8])

geweke<- gewekediag(post$yhat.train)

n<- length(geweke$train)

plot(geweke$z[1:100], pch='.', cex=2, ylab='z', xlab='i',
     sub=paste0('N:', n, ', k:', 1),
     xlim=c(0, n), ylim=c(-5, 5))
lines(1:n, rep(-1.96, n), type='l', col=6)
lines(1:n, rep(+1.96, n), type='l', col=6)
lines(1:n, rep(-2.576, n), type='l', col=5)
lines(1:n, rep(+2.576, n), type='l', col=5)
lines(1:n, rep(-3.291, n), type='l', col=4)
lines(1:n, rep(+3.291, n), type='l', col=4)
lines(1:n, rep(-3.891, n), type='l', col=3)
lines(1:n, rep(+3.891, n), type='l', col=3)
lines(1:n, rep(-4.417, n), type='l', col=2)
lines(1:n, rep(+4.417, n), type='l', col=2)
text(c(1, 1), c(-1.96, 1.96), pos=2, cex=0.6, labels='0.95')
text(c(1, 1), c(-2.576, 2.576), pos=2, cex=0.6, labels='0.99')
text(c(1, 1), c(-3.291, 3.291), pos=2, cex=0.6, labels='0.999')
text(c(1, 1), c(-3.891, 3.891), pos=2, cex=0.6, labels='0.9999')
text(c(1, 1), c(-4.417, 4.417), pos=2, cex=0.6, labels='0.99999')


