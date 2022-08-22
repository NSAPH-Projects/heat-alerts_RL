
library(rlang, lib.loc= "/n/home_fasse/econsidine/R/x86_64-pc-linux-gnu-library/")
library(torch)
torch::cuda_is_available() # TRUE if GPU is available
library(luz)
library(matrixStats)
library(dplyr)
0 # Selection for updating packages

setwd("/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL")

## Functions for Q-learning:

predict_nn<- function(model, newdata){ # if using gpus
  w<- as_array(model$parameters$lm1.weight$cpu())
  b<- as_array(model$parameters$lm1.bias$cpu())
  
  return(newdata %*% t(w) + b)
}

eval_Q<- function(S, Q_model){
  
  data<- model.matrix(~ A*., data.frame(A=0,S))
  Q0<- predict_nn(Q_model, data)
  
  data<- model.matrix(~ A*., data.frame(A=1,S))
  Q1<- predict_nn(Q_model, data)
  
  return(cbind(Q0, Q1))
}

choose_a<- function(Q_mat, iter){
  
  max.Q<- rowMins(Q_mat)
  argmax<- max.col(-Q_mat, ties.method = "first") - 1
  
  argmax[over_pos]<- 0
  max.Q[over_pos]<- Q_mat[over_pos, 1]
  
  print(iter)
  
  return(cbind(argmax, max.Q))
}

## Set up the data:

load("data/Train-Valid-Test.RData")

n_counties<- length(unique(Train$GEOID))
n_years<- 11
n_days<- 153

A<- Train[-seq(n_days, nrow(Train), n_days),"alert"]
R<- Train$N[-seq(n_days, nrow(Train), n_days)]
ep_end<- rep(c(rep(0,n_days-2),1),length(R)/(n_days-1))
gamma<- 0.999
Target<- R

States.1<- Train[, c("HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                     "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                     "BA_zone", "Pop_density", "Med.HH.Income",
                     "year", "dos", "holiday", # "Holiday", "dow",
                     "alert_sum")]

States<- States.1[-seq(n_days, nrow(Train), n_days),]
States.1<- States.1[-seq(1, nrow(Train), n_days),]

## Scale and one-hot encode:
S<- States %>% mutate_if(is.numeric, scale)
S.1<- States.1 %>% mutate_if(is.numeric, scale)

S_full<- model.matrix(~ A*., data.frame(A,S))

over_budget<- readRDS("data/Over_budget.rds")
original_rows<- 1:(n_days*n_years*2837)
Over_budget<- data.frame(over_budget, row=original_rows[-seq(n_days, length(original_rows), n_days)])
Over_budget<- Over_budget[which(Over_budget$row %in% 
                                  as.numeric(row.names(States))),] # adjust for training set
over_pos<- which(Over_budget$over_budget==1)


## Specify the model:

torch_manual_seed(321)

LM<- nn_module(
  
  initialize = function(){
    self$lm1 <- nn_linear(in_features = ncol(S_full), out_features = 1)
  },
  
  forward = function(x){
    x %>% self$lm1()
  }
)


## Train the model:
  ## have to do a lot of this manually for RL (instead of just training one NN)
  ## see Training section on https://blogs.rstudio.com/ai/posts/2020-09-29-introducing-torch-for-r/

# Just at the beginning:
model<- LM()
model$to(device = "cuda")

optimizer<- optim_adam(model$parameters)
optimizer$zero_grad()
l<- c()
iter<- 1
n<- length(R)

## Set up dataloader:
make_DS<- dataset(
  initialize = function(df){ # df is dataframe(Target, S)
    self$x<- torch_tensor(df[,-1])
    self$y<- torch_tensor(df[,1])
  },
  
  .getitem = function(i) {
    list(x = self$x[i, ], y = self$y[i])
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
)

S_ds<- make_DS(cbind(Target, matrix(as.numeric(S_full),ncol=ncol(S_full))))

b_size<- 30000
S_dl<- dataloader(S_ds, batch_size=b_size)

## Train the model:
s<- Sys.time()
coro::loop(for(b in S_dl){
  
  output<- model(b[[1]]$to(device = "cuda"))
  # break
  # s<- Sys.time()
  loss<- nnf_mse_loss(output, b[[2]]$to(device = "cuda"))
  loss$backward()
  optimizer$step()
  l<- c(l, loss$item())
  
  Q_mat<- eval_Q(S.1, model)
  
  AMQ<- choose_a(Q_mat, iter)
  
  Target<- R + gamma*(1-ep_end)*AMQ[,2]
  # break
  # e<- Sys.time()
  
  iter<- iter + 1
  
})
e<- Sys.time()
e-s

png("new_results/torch_lm_8-22.png")
plot(1:length(l), l)
dev.off()

saveRDS(model, "Aug_results/torch_lm_8-22.rds")
saveRDS(Target, "Aug_results/Target_8-22.rds")
saveRDS(l, "Aug_results/Loss_8-22.rds")
