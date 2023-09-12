
library(stringr)

truth<- read.csv("data/processed/Coef_sample.csv")
params<- unique(truth[,1])

truth_clean<- t(sapply(truth[,2], function(s){
  x<- strsplit(s, ", |\\(|\\)|\\[|\\], |\\],|,\n        |grad_fn=<ExpandBackward0>")[[1]]
  return(as.numeric(x[c(-1,-2, -length(x), -(length(x)-1))]))
}))
row.names(truth_clean)<- NULL
names(truth_clean)<- NULL

estimates<- read.csv("data/processed/Validation_coefs.csv")

use_ind<- 1:((length(params)+1)*500)
est_clean<- t(sapply(estimates[use_ind,2], function(s){as.numeric(strsplit(s, ", |\\[|\\]")[[1]][-1])}))
row.names(est_clean)<- NULL
names(est_clean)<- NULL

Coverage<- data.frame(matrix(0, nrow=length(params), ncol=3))
names(Coverage)<- c("CI_80", "CI_90", "CI_95")
for(p in params){
  vals<- truth_clean[which(truth[,1] == p),]
  CI_vals<- as.vector(unlist(est_clean[which(estimates[use_ind,1] == p),]))
  CI_80<- quantile(CI_vals, c(0.1, 0.9))
  CI_90<- quantile(CI_vals, c(0.05, 0.95))
  CI_95<- quantile(CI_vals, c(0.025, 0.975))
  Coverage[which(params == p),]<- c(mean(vals >= CI_80[1] & vals <= CI_80[2]),
           mean(vals >= CI_90[1] & vals <= CI_90[2]),
           mean(vals >= CI_95[1] & vals <= CI_95[2]))
}

Coverage<- round(Coverage,3)
Param<- params
Coverage<- cbind(Param, Coverage)
Coverage

colMeans(Coverage[,-1])
