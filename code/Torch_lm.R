
library(rlang, lib.loc= "/n/home_fasse/econsidine/R/x86_64-pc-linux-gnu-library/")
library(torch)
torch::cuda_is_available() # TRUE if GPU is available
library(luz)
library(matrixStats)
library(dplyr)
0 # Selection for updating packages

setwd("/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL")

## Functions for Q-learning:

# predict_nn<- function(model, newdata){ # if using gpus
#   
#   w<- as_array(model$parameters$lm1.weight$cpu())
#   b<- as_array(model$parameters$lm1.bias$cpu())
# 
#   return(newdata %*% t(w) + b)
# }

eval_Q<- function(Q_model, S_0, S_1){ # update this!
  
  Q0<- as_array(Q_model(S_0$to(device = "cuda"))$cpu())
  Q1<- as_array(Q_model(S_1$to(device = "cuda"))$cpu())
  
  # Q0<- predict_nn(Q_model, S_0)
  # Q1<- predict_nn(Q_model, S_1)
  
  return(cbind(Q0, Q1))
}

choose_a<- function(Q_mat, over, iter){
  
  max.Q<- rowMins(Q_mat)
  argmax<- max.col(-Q_mat, ties.method = "first") - 1
  
  argmax[over]<- 0
  max.Q[over]<- Q_mat[over, 1]
  
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
S.1_full_0<- model.matrix(~ A*., data.frame(A=0,S))
S.1_full_1<- model.matrix(~ A*., data.frame(A=1,S))

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

## Set up dataloader:
make_DS<- dataset(
  initialize = function(df){ # df is list(Index, S_full, S.1_full_0, S.1_full_1, 
                                                                      # R, ep_end)
    self$index<- torch_tensor(df[[1]])
    self$s<- torch_tensor(df[[2]])
    self$s.1_0<- torch_tensor(df[[3]])
    self$s.1_1<- torch_tensor(df[[4]])
    self$r<- torch_tensor(df[[5]])
    self$ee<- torch_tensor(df[[6]])
    
  },
  
  .getitem = function(i) {
    list(index = self$index[i], s = self$s[i,], s.1_0 = self$s.1_0[i,],
         s.1_1 = self$s.1_1[i,], r = self$r[i], ee = self$ee[i])
  },
  
  .length = function() {
    self$r$size()[[1]]
  }
)

Index<- 1:length(Target)
S_ds<- make_DS(list(Index, matrix(as.numeric(S_full),ncol=ncol(S_full)),
                     matrix(as.numeric(S.1_full_0),ncol=ncol(S_full)), 
                     matrix(as.numeric(S.1_full_1),ncol=ncol(S_full)),
                     R, ep_end))

b_size<- 256
S_dl<- dataloader(S_ds, batch_size=b_size)

## Train the model:

model<- LM()
model$to(device = "cuda")

optimizer<- optim_adam(model$parameters)
l<- c()
iter<- 1
n<- length(R)

new_coefs<- as_array(model$parameters$lm1.weight$cpu())
old_coefs<- rep(0, length(new_coefs))

Coefs<- new_coefs
L<- c()
K<- 1

s<- Sys.time()
while(sqrt(mean((new_coefs - old_coefs)^2)) > 0.1 | iter== 1){
  coro::loop(for(b in S_dl){
    optimizer$zero_grad()
    output<- model(b$s$to(device = "cuda"))
    # break
    # s<- Sys.time()
    
    inds<- as_array(b$index)
    
    target<- torch_tensor(Target[inds])
    
    loss<- nnf_mse_loss(output, target$to(device = "cuda"))
    loss$backward()
    optimizer$step()
    l<- c(l, loss$item())
    
    with_no_grad({
      Q_mat<- eval_Q(model, b$s.1_0, b$s.1_1)
      
      o<- which(inds %in% over_pos)
      
      AMQ<- choose_a(Q_mat, o, iter)
      
      Target[inds]<- as_array(b$r$cpu()) + gamma*(1-as_array(b$ee$cpu()))*AMQ[,2]
      # break
    })
    
    # e<- Sys.time()
    
    iter<- iter + 1
    
  })
  
  # optimizer$zero_grad()
  # output<- model(torch_tensor(matrix(as.numeric(S_full),ncol=ncol(S_full)))$to(device = "cuda") )
  # 
  # loss<- nnf_mse_loss(output, torch_tensor(Target)$to(device = "cuda"))
  # loss$backward()
  # optimizer$step()
  # L<- c(L, loss$item())
  
  with_no_grad({
    Q_mat<- eval_Q(model, torch_tensor(matrix(as.numeric(S.1_full_0),ncol=ncol(S_full))),
                   torch_tensor(matrix(as.numeric(S.1_full_1),ncol=ncol(S_full))))
    
    AMQ<- choose_a(Q_mat, over_pos, iter)
    
    Target<- R + gamma*(1-ep_end)*AMQ[,2]
  })
  
  old_coefs<- new_coefs
  new_coefs<- as_array(model$parameters$lm1.weight$cpu())
  
  Coefs<- rbind(Coefs, new_coefs)
  
  print(paste("Finished Epoch:", K))
  K<- K+1
  break
}
e<- Sys.time()
e-s

png("new_results/torch_lm_8-30.png")
plot(1:length(l), l)
dev.off()

saveRDS(model, "Aug_results/torch_lm_8-30.rds")
saveRDS(Target, "Aug_results/Target_8-30.rds")
saveRDS(l, "Aug_results/Loss_8-30.rds")
saveRDS(Coefs, "Aug_results/Q-coefficients_8-30.rds")

