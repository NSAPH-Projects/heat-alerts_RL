
library(stringr)

setwd("C:/Users/ellen/OneDrive/MyDocs/Graduate Research/Heat alerts RL")

## Data from https://zenodo.org/record/7141231#.Y6RzxXbMLrc
data<- read.csv("CHENlab-Yale-HVI_US-6d51d3e/Heat Vulnerability and Redlining Data GeoHealth Journal_1003.csv")

geoid<- str_pad(data$GEOID, 11, pad="0")

fips<- substring(geoid, 1, 5)

DF<- data.frame(fips, Population=rowSums(data[,13:20]),
                HVI_score=data$HVI_Score, 
                Pct_elderly_alone=data$Percentage_Population_Elderly_and_Living_Alone)

f<- unique(fips)
pos<- which(DF$fips == f[1])
HVI_score<- weighted.mean(DF$HVI_score[pos], DF$Population[pos])
Pct_elderly_alone<- weighted.mean(DF$Pct_elderly_alone[pos], DF$Population[pos])
for(i in 2:length(f)){
  pos<- which(DF$fips == f[i])
  HVI_score<- append(HVI_score, weighted.mean(DF$HVI_score[pos], DF$Population[pos]))
  Pct_elderly_alone<- append(Pct_elderly_alone, weighted.mean(DF$Pct_elderly_alone[pos], DF$Population[pos]))
}

Data<- data.frame(fips=f, HVI_score, Pct_elderly_alone)

write.csv(Data, "HVI_county-level.csv", row.names=FALSE)

