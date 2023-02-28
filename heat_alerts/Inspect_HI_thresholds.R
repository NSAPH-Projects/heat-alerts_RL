
library(dplyr)

load("data/Train-Test.RData")

Train<- Train[which(Train$Population>= 65000),]

## Cutoff by region
for(z in unique(Train$BA_zone)){
  pos<- which(Train$BA_zone == z)
  print(z)
  print(quantile(Train$HImaxF_PopW[pos], seq(0.85, 1, 0.025)))
}

for(z in unique(Train$BA_zone)){
  pos<- which(Train$BA_zone == z & Train$alert == 1)
  print(z)
  print(min(Train$HImaxF_PopW[pos]))
}

## Cutoff by county

fips<- unique(Train$fips)
pct90<- c()
for(i in fips){
  pos<- which(Train$fips == i)
  pct90<- append(pct90, quantile(Train$HImaxF_PopW[pos], 0.9))
  print(i)
}

summary(pct90)
hist(pct90)

county_alerts<- c()
for(i in fips){
  pos<- which(Train$fips == i & Train$alert == 1)
  if(length(pos) == 0){
    county_alerts<- append(county_alerts, NA)
  }else{
    county_alerts<- append(county_alerts, min(Train$HImaxF_PopW[pos]))
  }
  print(i)
}

summary(county_alerts)
hist(county_alerts)


