
library(spam, lib.loc = "~/apps/R_4.2.2")
library(fields, lib.loc = "~/apps/R_4.2.2")
library(dplyr)
library(ape, lib.loc = "~/apps/R_4.2.2")

# load("data/Small_S-A-R_prepped.RData")
# R<- R_other_hosps[which((DF$quant_HI_county*qhic_sd + qhic_mean) >= 0.9),1]
# rm(list=setdiff(ls(), "R"))

# fips<- Top3rd$fips[which(Top3rd$quant_HI_county >= 0.9)] # Top3rd from Data_for_Python.R
# saveRDS(fips, "Summer_results/Pct90_fips.rds")

# days<- Top3rd$dos[which(Top3rd$quant_HI_county >= 0.9)] # Top3rd from Data_for_Python.R
# saveRDS(days, "Summer_results/Pct90_dos.rds")

fips<- readRDS("Summer_results/Pct90_fips.rds")
dos<- readRDS("Summer_results/Pct90_dos.rds")

## Get average of coords for each county:

coords<- readRDS("data/County_coords.rds")

coords$long<- as.numeric(coords$long)
coords$lat<- as.numeric(coords$lat)

train_coords<- coords[which(coords$set == "Train"),]

## Make inverse-distance matrix:
a<- RdistEarth(as.matrix(train_coords[,c("long","lat")]), miles = TRUE)
a[which(a==0)]<-1

# A<- 1/(a*a)
A<- 1/a
diag(A)<- 0
# summary(as.vector(A))

#### Calculate Moran's I:

Morans<- function(x, D){
  # x_bar<- mean(x)
  # x_diff<- x - x_bar
  # N<- length(x)
  # W<- sum(D)
  # 
  # denom<- sum(x_diff^2)
  # numer<- sum(D * (x_diff %o% x_diff))
  # I<- (N / W)*(numer / denom)
  
  y<- Moran.I(x, D, scaled = TRUE, alternative = "two.sided") # greater?
  return(y$observed)
}

R<- readRDS("Summer_results/Kaggle_preds_a.rds")[,2]
  
preds<- readRDS("Summer_results/Kaggle_preds_a.rds")[,1]
cor(R, preds)^2
1 - (sum((R-preds)^2)/(((length(preds)-1)*var(R))))

preds<- readRDS("Summer_results/Kaggle_preds_b.rds")
cor(R, preds)^2
1 - (sum((R-preds)^2)/(((length(preds)-1)*var(R))))

resids<- R - preds

## Calculate for each day, then average across days:

MI<- c()
s1<- c()
s2<- c()
s3<- c()
s4<- c()
s5<- c()
n<- c()
w<- c()

for(d in sort(unique(dos))){
  day_pos<- which(dos == d)
  k<- length(day_pos)
  if(k != 1){
    locs<- fips[day_pos]
    m<- match(locs, train_coords$fips)
    x<- resids[day_pos]
    df<- data.frame(m,x)
    df<- aggregate(x ~ m, df, mean)
    D<- A[df$m, df$m]
    x<- df$x
    MI<- append(MI, Morans(x, D))
    s1<- append(s1, 0.5*sum((2*D)^2))
    s2<- append(s2, sum((rowSums(D) + colSums(D))^2))
    z<- x - mean(x)
    q<- length(x)
    s3<- append(s3, (1/q)*sum(z^4)/((1/q)*sum(z^2))^2)
    s4<- append(s4, (q^2 - 3*q + 3)*s1[length(s1)] - q*s2[length(s2)] + 3*sum(D)^2)
    s5<- append(s5, (q^2 - q)*s1[length(s1)] - 2*q*s2[length(s2)] + 6*sum(D)^2)
    n<- append(n, q)
    w<- append(w, sum(D))
  }
}

e<- mean(-1/(n-1))
o<- mean(MI)
n_bar<- mean(n)
v<- (n_bar*mean(s4) - mean(s3)*mean(s5))/((n_bar - 1)*(n_bar-2)*(n_bar-3)*mean(w)^2) - e^2
pnorm(o, e, sqrt(v)) # can't do this bc v is negative


e<- weighted.mean(-1/(n-1), w = n)
o<- weighted.mean(MI, w = n)
v<- (n_bar*weighted.mean(s4, w=n) - weighted.mean(s3, w=n)*weighted.mean(s5, w=n))/((n_bar - 1)*(n_bar-2)*(n_bar-3)*weighted.mean(w,w=n)^2) - e^2
pnorm(o, e, sqrt(v))


## Calculate autocorrelation across days for each location, then average across locations:

Acl<- c()
n_days<- c()

for(f in sort(unique(fips))){
  fips_pos<- which(fips == f)
  k<- length(fips_pos)
  if(k != 1){
    days<- dos[fips_pos]
    x<- resids[fips_pos]
    df<- data.frame(days, x)
    df<- aggregate(x ~ days, df, mean)
      
    diffs<- df$days[-1]-df$days[-nrow(df)]
    ts<- df$x[1]
    for(i in 1:length(diffs)){
      if(diffs[i] > 1){
        ts<- append(ts, rep(NA, diffs[i]-1))
      }
      ts<- append(ts, df$x[i+1])
    }
    
    ac<- acf(ts, na.action=na.pass, plot = FALSE)$acf
    # ac1_se<- sqrt((1+2*ac[2]^2)/length(ts))  # https://stats.stackexchange.com/questions/368404/confidence-intervals-for-autocorrelation-function
    
    Acl<- append(Acl, ac[2])
    n_days<- append(n_days, length(ts))
  }
}

AC1<- mean(Acl)
# AC1<- weighted.mean(Acl, w=n_days) # barely different

ac1_se<- sqrt((1+2*AC1^2)/mean(n_days))

1 - pnorm(AC1, 0, ac1_se)


#### 


