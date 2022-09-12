

## Get J and x=w*R from OPE function...

ID<- rep(1:(length(x)/(n_days-1)), each = (n_days-1)) 

X<- aggregate(x ~ ID, data = data.frame(ID, X), sum)[,2]

#### Calculate lower bound given confidence level d:

N<- length(X)

set.seed(321)
pre<- sample(1:N, round(0.05*N), replace = FALSE)
post<- which(!1:N %in% pre)
n_pre<- length(pre)
n_post<- N - n_pre


c_fun<- function(c, d){
  
  Y<- X[pre]
  Y[which(Y > c)]<- c
  
  s<- mean(Y) - 7*c*log(2/d)/(3*(N - n_pre - 1)) - sqrt((log(2/d)/n_post)*4*var(Y))
  
  return(-s) # so optim can minimize 
}


d<- 0.05

c<- optim(1, c_fun, lower = 0, method = "L-BFGS-B", d = d)$par

Y<- X[post]
Y[which(Y > c)]<- c

(n_post/c)*( sum(Y)/c - 7*n_post*log(2/d)/(3*(n_post-1))
             - sqrt(2*log(2/d)*2*n_post*var(Y/c)) )

# (n_post/c)*( sum(Y)/c - 7*n_post*log(2/d)/(3*(n_post-1)) 
#              - sqrt((2*log(2/d)/(n_post-1))*(n*sum((Y/c)^2) - sum(Y/c)^2)) )


#### Find minimum d:

d_fun<- function(d, Jb){
  
  c<- optim(1, c_fun, lower = 0, method = "L-BFGS-B", d = d)$par
  
  Y<- X[post]
  Y[which(Y > c)]<- c
  
  k1<- 7*n_post/(3*(n_post-1)) 
  k3<- Jb*n_post/c - sum(Y)/c
  
  k2<- sqrt( 4*n_post*var(Y/c) )
  z<- (-k2 + sqrt(k2^2 - 4*k1*k3))/(2*k1)
  
  # Calculate 1-d:
  if(!is.na(z) & z > 0){
    omd<- 1 - min(1, 2*exp(-z^2))
  }else{
    omd<- 0 
  }

  return(1 - omd) # returning d
}


D<- optim(0.5, d_fun, lower = 0, upper = 1, method = "L-BFGS-B", Jb = -1761)





  
  
