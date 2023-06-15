
library(spam, lib.loc = "~/apps/R_4.2.2")
library(fields, lib.loc = "~/apps/R_4.2.2")
library(dplyr)

## Get average of coords for each county:
coords<- readRDS("data/County_coords.rds")
ctrd<- aggregate(. ~ fips, coords[,1:3], mean)
Ctrd<- inner_join(ctrd, distinct(coords[,c("fips", "set")]))

Ctrd$long<- as.numeric(Ctrd$long)
Ctrd$lat<- as.numeric(Ctrd$lat)

## Make inverse-distance matrix:
a<- RdistEarth(as.matrix(Ctrd[which(Ctrd$set == "Train"),c("long","lat")]), miles = TRUE)
a[which(a==0)]<-1

# A<- 1/(a*a)
A<- 1/a
summary(as.vector(A))

## Calculate Moran's I:

MI<- function(x){
  
}


