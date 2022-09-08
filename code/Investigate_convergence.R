

#### Torch lm:

Coefs<- readRDS("Aug_results/Q-coefficients_9-7.rds")

## Looking at coefs all together:

c0<- Coefs[1:(nrow(Coefs)-1),]
c1<- Coefs[2:nrow(Coefs),]

res<- apply(cbind(c0, c1), MARGIN = 1, function(x) sqrt(mean((x[49:96] - x[1:48])^2)))

plot(1:length(res), res)

## Now looking at the individual coefs:

cols<- readRDS("data/Model_colnames_9-8.rds")
colnames(Coefs)<- cols

par(mfrow=c(3,4))
for(j in 1:ncol(Coefs)){
  plot(1:nrow(Coefs), Coefs[,j], main = cols[j])
}

##########################################################

## Read in text file:
res<- read.table("Aug_results/First_SGD.txt", fill=TRUE)

## Spread in coefficients:
diff_coefs<- res[seq(2,nrow(res),28),2]
plot(1:999, diff_coefs, type = "l")

## Value of intervention, all else equal:
A_coef<- res[seq(6,nrow(res), 28),2]
plot(1:999, A_coef)

## Effect of alert_sum:
alert_sum<- res[seq(18,nrow(res), 28),1]
plot(1:999, alert_sum)

## Effect of Med HH Income:
med_HH<- res[seq(14,nrow(res), 28),3]
plot(1:999, med_HH)
