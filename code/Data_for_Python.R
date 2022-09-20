library(readr)

load("data/Train-Test.RData")

over_budget<- readRDS("data/Over_budget.rds")
n_days<- 153
n_years<- 11
original_rows<- 1:(n_days*n_years*2837)
Over_budget<- data.frame(over_budget, row=original_rows[-seq(n_days, length(original_rows), n_days)])

write_csv(Over_budget, "data/Over_budget_S.csv")

S_inds<- as.numeric(row.names(Train[-seq(n_days, nrow(Train), n_days),]))
Top3rd<- Train[which(Train$Population>= 65000),]
S_t3_inds<- as.numeric(row.names(Top3rd[-seq(n_days, nrow(Top3rd), n_days),]))

write_csv(data.frame(S_inds), "data/S_training_indices.csv")
write_csv(data.frame(S_t3_inds), "data/S_t3_training_indices.csv")

Over_budget_train<- Over_budget[which(Over_budget$row %in% S_inds),]
write_csv(Over_budget_train, "data/Over_budget_S_train.csv")

Over_budget_t3<- Over_budget[which(Over_budget$row %in% S_t3_inds),]
write_csv(Over_budget_t3, "data/Over_budget_S_t3.csv")


