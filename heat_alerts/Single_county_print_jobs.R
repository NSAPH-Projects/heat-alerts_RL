
## All the different options:

algos<- c("DQN", "DoubleDQN", "CPQ")
MR<- c("True", "False")
elig<- c("all", "90pct")
seeds<- c("321", "221", "121")
fips<- c("4013", "6085", "36061")
NHU<- c(16, 32, 64, 128, 256)
NHL<- c(1, 2, 3)
LR<- c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)
SR<- c(1, 2, 3, 5, 10)
b_size<- c(16, 32, 64, 128)
ma<- c(20, 50, 100, 200)

## Preliminary:
algos<- c("DoubleDQN", "CPQ")
MR<- c("True", "False")
seed<- c("321")
fips<- c("4013")
NHU<- c(16, 32, 64, 128)
NHL<- c(2, 3)
LR<- c(0.0001, 0.001, 0.01)
SR<- c(3)
b_size<- c(32)
ma<- c(20)

tests<- expand.grid(algos, MR, seed, fips, NHU, NHL, LR, SR, b_size, ma)


