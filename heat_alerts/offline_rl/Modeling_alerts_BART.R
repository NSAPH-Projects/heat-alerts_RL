
library(BART)
# library(profvis)
library(ggplot2)
library(parallel)
detectCores()

## Read in data and make dummy vars:

n_days<- 153

load("data/Small_S-A-R_prepped.RData")

DF$alert<- A


## Set up train and "test" sets:

set.seed(321)
train_inds<- sample(1:nrow(DF), 0.75*nrow(DF), replace = FALSE)

alert_pos<- which(names(DF) == "alert")
x.train<- DF[train_inds, -alert_pos]
y.train<- DF[train_inds, "alert"]

test_inds<- setdiff(1:nrow(DF), train_inds)

x.test<- DF[test_inds, -alert_pos]
y.test<- DF[test_inds, "alert"]


## Set up BART:

M<- 5000 # number of posterior samples (1000 is the default, after 100 burn-in)
B<- 10 # number of cores (on the cluster)

start<- Sys.time()
# post<- pbart(x.train, y.train, x.test, 
             # ndpost = 100, nskip = 10, printevery = 10)
post<- mc.pbart(x.train, y.train, printevery = 10, ndpost = 100, 
             nskip=5000, keepevery = 1000, mc.cores = 5, seed = 321)
# post<- mc.pbart(x.train, y.train, x.test, ndpost = M,
#                 mc.cores = B, seed = 321,
#                 printevery = 10)
p<- list(treedraws = post$treedraws, binaryOffset = post$binaryOffset)
class(p)<- "pbart"
saveRDS(p, "Fall_results/BART-model_12-29.rds")
end<- Sys.time()
end - start
## If we use the argument "sparse=TRUE", then can look at post$varprob...

## Look at the posterior samples: post$yhat.train and post$yhat.test
  ## If using pbart, look at prob.train or prob.test to get probabilities (pnorm of yhats)

preds.train<- post$prob.train
preds.a1_train<- preds.train[,which(y.train == 1)]
preds.a0_train<- preds.train[,which(y.train == 0)]

## If looking later:
p<- readRDS("Fall_results/BART-model_11-20.rds")

x.train$holiday1<- 1
x.train$holiday1[x.train$holiday == 1]<- 0
x.train$holiday2<- 0
x.train$holiday2[x.train$holiday == 1]<- 1

x.test$holiday1<- 1
x.test$holiday1[x.test$holiday == 1]<- 0
x.test$holiday2<- 0
x.test$holiday2[x.test$holiday == 1]<- 1

### Looking at a=0 and a=1 separately:

# a1_train<- sample(which(y.train == 1),1000)
# a1_test<- sample(which(y.test == 1),1000)
a1_train<- which(y.train == 1)
a1_test<- which(y.test == 1)

preds.a1_train<- predict(p, x.train[a1_train,names(p$treedraws$cutpoints)])$prob.test
# s<- Sys.time()
preds.a1_test<- predict(p, x.test[a1_test,names(p$treedraws$cutpoints)])$prob.test
# e<- Sys.time()

# a0_train<- sample(which(y.train == 0),1000)
# a0_test<- sample(which(y.test == 0),1000)
a0_train<- which(y.train == 0)
a0_test<- which(y.test == 0)

preds.a0_train<- predict(p, x.train[a0_train,names(p$treedraws$cutpoints)])$prob.test
# s<- Sys.time()
preds.a0_test<- predict(p, x.test[a0_test,names(p$treedraws$cutpoints)])$prob.test
# e<- Sys.time()

#### Check convergence:

check_conv<- function(preds, obs=NULL, a1=FALSE, a0=FALSE){
  
  set.seed(321)
  
  ## Trace plots:
  if(a1==TRUE){
    i<- sample(1:ncol(preds),10)
  }else if(a0==TRUE){
    i<- sample(1:ncol(preds),50)
  }else{
    i<- c(sample(which(obs == 0),3), sample(which(obs == 1),7))
  }
  y<- max(preds[,i])
  for(j in 1:length(i)) {
    if(j==1){
      plot(preds[ , i[j]],
           type='l', ylim=c(0, y),
           # sub=paste0('N:', n, ', k:', k),
           ylab=expression(Phi(f(x))), xlab='m',
           main = "Trace", col=j)
    }else{
      lines(preds[ , i[j]],
            type='l', col=j)
    }
  }
    
    ## Checking autocorrelation:
    
    auto.corr<- acf(preds[ , i], plot=FALSE)
    max.lag<- max(auto.corr$lag[ , 1, 1])
    
    j <- seq(-0.5, 0.4, length.out=10)
    for(h in 1:length(i)) {
      if(h==1)
        plot(1:max.lag+j[h], auto.corr$acf[1+(1:max.lag), h, h],
             type='h', xlim=c(0, max.lag+1), ylim=c(-1, 1),
             ylab='acf', xlab='lag', main = "Autocorrelation",col=h)
      else
        lines(1:max.lag+j[h], auto.corr$acf[1+(1:max.lag), h, h],
              type='h', col=h)
    }
    
    ## Geweke statistic:
    
    if(a1==TRUE | a0==TRUE){
      k<- sample(1:ncol(preds), 1000)
    }else{
      k<- c(sample(which(obs == 0),500), sample(which(obs == 1),500))
    }
    
    geweke<- gewekediag(preds[,k])
    n<- length(geweke$z)
    
    plot(geweke$z, pch='.', cex=2, ylab='z', xlab='i',
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
  
}

## Inspect:

check_conv(preds.train, y.train)
# check_conv(preds.train[seq(1,nrow(preds.train),2),], y.train)

check_conv(preds.a1_train, a1=TRUE)
check_conv(preds.a0_train, a0=TRUE)

check_conv(preds.a1_test, a1=TRUE)
check_conv(preds.a0_test, a0=TRUE)

## All_preds:

all_probs<- cbind(preds.a1_train, preds.a0_train, preds.a1_test, preds.a0_test)

probs_a0<- cbind(preds.a0_train, preds.a0_test)
mean(probs_a0 >= 0.01) # 0.0553

means_a0<- colMeans(probs_a0)
mean(means_a0 >= 0.01) # 0.0601

