
load("data/Train-Test.RData")

hist(Train$T_since_alert)

hist(Train$T_since_alert[Train$alert==1])
