
W<- read.csv("data/Final_30_W.csv")

get_hosps<- function(x){ # where x is a vector from avg_return function
  y<- (1 - x)*W$Offset
  return(y*152) # per summer
}

## Apply:

bench<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
bench<- bench[match(W$Fips, bench$County),]

nohr<- apply(bench[,c("Zero", "NWS", "Random", "Top_K", "Random_QHI", "AA_QHI")],
      MARGIN=2, get_hosps)

compared_to_zero<- apply(nohr[,-1], MARGIN=2, function(y){nohr[,1]-y})
compared_to_nws<- apply(nohr[,3:ncol(nohr)], MARGIN=2, function(y){nohr[,2]-y})

colSums(compared_to_zero)
colSums(compared_to_zero)*11
colMeans(compared_to_zero)
apply(compared_to_zero, MARGIN=2, median)

colSums(compared_to_nws)
colSums(compared_to_nws)*11
colMeans(compared_to_nws)
apply(compared_to_nws, MARGIN=2, median)


