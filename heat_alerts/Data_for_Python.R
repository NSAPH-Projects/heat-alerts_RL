library(readr)

load("data/Train-Test.RData")

S_inds<- as.numeric(row.names(Train[-seq(n_days, nrow(Train), n_days),]))
Top3rd<- Train[which(Train$Population>= 65000),]
S_t3_inds<- as.numeric(row.names(Top3rd[-seq(n_days, nrow(Top3rd), n_days),]))

# write_csv(data.frame(S_inds), "data/S_training_indices.csv")
# write_csv(data.frame(S_t3_inds), "data/S_t3_training_indices.csv")

# write_csv(Top3rd, "data/Train_smaller-for-Python.csv")


