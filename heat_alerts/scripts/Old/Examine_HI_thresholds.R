
library(readr)

data<- read_csv("data/Summer23_Train_smaller-for-Python.csv")

counties<- c("41067", "53015", "20161", "37085", "48157", 
           "28049", "19153", "17167", "31153", "06071", "04013")

for(k in counties){
  d<- data[which(data$fips == k),]
  print(k)
  print(round(as.vector(quantile(d$HImaxF_PopW, seq(0.1, 0.9, 0.1))), 1))
}

