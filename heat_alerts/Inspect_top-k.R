
library(readr)
library(ggplot2)

data<- read_csv("data/Summer23_Train_smaller-for-Python.csv")

pred_R<- read.csv("Summer_results/R_6-28_forced_small-S_all.csv")

#### First, for a few key counties:

# fips<- "04013"
# fips<- "06085"
# fips<- "36061" 
fips<- unique(data$fips)

beneficial<- c()
adverse<- c()
b.rate<- c()
b.total<- c()
b.Pop<- c()
n.rate<- c()
n.total<- c()
n.Pop<- c()
for(f in fips){
  county<- data[which(data$fips %in% f),]
  county$Day<- 1:nrow(county)
  
  # ggplot(county, aes(x=Day)) +
  #   geom_point(aes(y=quant_HI_county, color = as.factor(1-alert)),
  #              show.legend = FALSE) +
  #   geom_line(aes(y=1000*other_hosps / total_count)) +
  #   ylab("Quantile of HI (col=alert); NOHR/1,000") +
  #   ggtitle(paste("County", fips))
  
  budgets<- county$alert_sum[which(county$dos == 153)]
  pred_R_county<- pred_R[which(data$fips %in% f),]
  diffs_county<- pred_R_county$X1 - pred_R_county$X0
  
  # plot(county$dos, diffs_county, main = paste("County", f),
  #      xlab = "Day of Summer", ylab = "Modeled R1-R0")
  
  # ggplot(data.frame(county, diffs_county),
  #        aes(x=dos, y = diffs_county, color = ""))
  
  ## All days:
  
  NWS<- pred_R_county$X0
  NWS[county$alert == 1]<- pred_R_county$X1[county$alert == 1]
  
  years<- 2006:2015
  n_years<- length(years)
  
  topK<- c()
  
  for(y in 1:n_years){
    y_pos<- which(county$year == years[y])
    b<- budgets[y]
    a<- order(diffs_county[y_pos], decreasing = TRUE)[1:b]
    t<- pred_R_county$X0[y_pos]
    if(b > 0){
      t[a]<- pred_R_county$X1[y_pos][a]
    }
    topK<- append(topK, t)
    # print(years[y])
    # print(sum(t))
    # print(sum(NWS[y_pos]))
  }
  
  # summary(topK - NWS)
  # sum(topK - NWS)
  g<- mean(county$total_count)*sum(topK - NWS)/1000
  if(g > 0){
    beneficial<- append(beneficial, f)
    b.rate<- append(b.rate, sum(topK-NWS))
    b.total<- append(b.total, g)
    b.Pop<- append(b.Pop, county$Population[1])
  }else{
    adverse<- append(adverse, f)
    n.rate<- append(n.rate, sum(topK-NWS))
    n.total<- append(n.total, g)
    n.Pop<- append(n.Pop, county$Population[1])
  }
}

b.DF<- data.frame(fips = beneficial, Rate = b.rate,
                  Total = b.total, Population = b.Pop)
n.DF<- data.frame(fips = adverse, Rate = n.rate, 
                  Total = n.total, Population = n.Pop)

b.DF[order(b.DF$Rate, decreasing = TRUE),]
b.DF[order(b.DF$Total, decreasing = TRUE),]
b.DF[order(b.DF$Population, decreasing = TRUE),]
rate<- c(b.rate, n.rate)
hist(rate)


## Top 90pct HI days:

f<- "29510"
f<- "36047" # Kings County, NY
f<- "06037"

county<- data[which(data$fips %in% f),]
pred_R_county<- pred_R[which(data$fips %in% f),]
diffs_county<- pred_R_county$X1 - pred_R_county$X0

plot(county$dos, diffs_county, main = paste("County", f),
     xlab = "Day of Summer", ylab = "Modeled R1-R0")

HI_pos<- which(county$quant_HI_county >= 0.9)
NWS<- pred_R_county$X0[HI_pos]
county_HI<- county[HI_pos,]
NWS[county_HI$alert == 1]<- pred_R_county$X1[HI_pos][county_HI$alert == 1]

years<- 2006:2015
n_years<- length(years)

topK<- c()

for(y in 1:n_years){
  y_pos<- which(county_HI$year == years[y])
  b<- budgets[y]
  t<- pred_R_county$X0[HI_pos][y_pos]
  if(b > length(y_pos)){
    a<- 1:length(y_pos)
    print("b > y_pos")
  }else{
    a<- match(1:b, order(diffs_county[HI_pos][y_pos]))
  }
  if(b > 0){
    t[a]<- pred_R_county$X1[HI_pos][y_pos][a]
  }
  topK<- append(topK, t)
}

summary(topK - NWS)
sum(topK - NWS)
mean(county$total_count)*sum(topK - NWS)/1000



