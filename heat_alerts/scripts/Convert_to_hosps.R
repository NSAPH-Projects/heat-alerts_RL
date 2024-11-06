
W<- read.csv("data/Final_30_W.csv")

get_hosps<- function(x){ # where x is a vector from avg_return function
  y<- (1 - x/152)*W$Offset/W$total_count
  return(y*152) # per summer
}

all_medicare<- 49000000


