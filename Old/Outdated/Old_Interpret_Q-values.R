
#### Sourcing everything from OPE.R...

## Here are the weights:

y_100.qt_0025<- c(4.9002988, 0.9561502, 122.8304463, -467.6929355, 
                  -626.8763427, 5.3314790, 0.5666687, 125.3147654, 
                  -467.0453364, -616.1291645)
y_100.qt_002<- c(3.227026, -1.057144, 157.743203, -397.232099,
                  -915.638060, 2.3019162, -0.1524963, 158.0077266,
                  -396.6269728, -903.3962598)
y_100.qt_00175<- c(2.0424366, -2.5993480, 241.4489724, -630.5807619,
                   -1430.6157089, 0.6076141, -0.6964970, 241.2610769,
                   -629.8293962, -1415.5634076)
y_100.qt_0015<- c(1.0884783, -4.2786934, 342.4826609, -827.0148787,
                  -1696.9958081, -0.7456498, -1.4345054, 342.1957501,
                  -826.1844939, -1679.2944142)
y_100.qt_001<- c(0.5592941, -5.3593660, 477.8782490, -1710.4075695,
                 -635.3351123, -1.0490838, -1.7841658, 477.4670577,
                 -1708.6196682,  -615.0188748)
y_100.qt_0005<- c(4.357196, -5.755611, 603.492995, -2671.301377, 251.558483,
                  4.136001, -1.775227, 602.964898, -2667.975966, 274.604229)

for(w in list( # y_100.qt_0025, 
              y_100.qt_002, y_100.qt_00175,  
           y_100.qt_0015, y_100.qt_001, y_100.qt_0005)){
  W<- data.frame(No_alert=w[1:ncol(S)], Alert = w[(ncol(S) + 1):(2*ncol(S))])
  row.names(W)<- colnames(S)
  print(W)
}

## Make a plot:

library(ggplot2)

q_tols<- c(0.0025, 0.002, 0.00175, 0.0015, 0.001, 0.0005)
weights<- list(y_100.qt_0025, y_100.qt_002, y_100.qt_00175, y_100.qt_0015, 
               y_100.qt_001, y_100.qt_0005)
Num_alerts<- rep(0, 5)

for(i in 1:5){

  w<- weights[[i]]
  q_tol<- q_tols[i]

  s<- ncol(S_test)
  Q0<- S_test%*%w[1:s]
  Q1<- S_test%*%w[(s + 1):(2*s)]
  # Q0<- S.1[train,]%*%w[1:s]
  # Q1<- S.1[train,]%*%w[(s + 1):(2*s)]
  q_scale<- sd(c(Q0, Q1))
  Q<- cbind(Q0, Q1, Q1-Q0)/q_scale
  policy<- rep(0, nrow(Q))
  policy[which(Q[,3] > 0 & Q[,3]/abs(Q[,1]) > q_tol)]<- 1

  Num_alerts[i]<- sum(policy)
}

Num_alerts / N

Num_alerts<- c(138632, 176822, 254348, 345159, 422224, 612292) # from the training results

J_1<- c(-91.86163, -87.37957, -87.43654, -86.28342, -89.0691, -90.25862)
J_0<- c(-117.8713, -89.28409, -89.12207, -84.8756, -89.86042, -90.33333)
J_2<- c(-90.16397, -85.68192, -85.7404, -84.58577, -87.37144, -88.56096)

plot_DF<- data.frame(q_tols, J_0, J_1, J_2, Num_alerts)

ggplot(plot_DF, aes(x = q_tols, y = J_0, color = "S rounded to 0 decimals")) + geom_line() +
  geom_line(aes(y = J_1, color = "S rounded to 1 decimal")) + 
  geom_line(aes(y = J_2, color = "S rounded to 2 decimals")) +
  scale_colour_manual("", 
                      breaks = c("S rounded to 0 decimals", 
                                 "S rounded to 1 decimal", 
                                 "S rounded to 2 decimals"),
                      values = c("red", "green", "blue")) + 
  xlab("Tolerance for Percent Difference in Q Values") +
  ylab("OPE value (J)")


## Best weights:

w<- y_100.qt_0015

## Looking at Q values for combos of my choice (on the evaluation set):







