
library(ggplot2)

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

Regions<- c("Cold", "Hot Dry", "Hot Humid", "Mixed Humid", "Marine")
n_days<- 153

Data<- read.csv(paste0("heat_alerts/time_series_data/HI-lam-tau_", "Cold", ".csv"))[,-1]
Data$Region<- "Cold"
nrow(Data)
for(r in Regions[-1]){
  data<- read.csv(paste0("heat_alerts/time_series_data/HI-lam-tau_", r, ".csv"))[,-1]
  data$Region<- r
  Data<- rbind(Data, data)
  print(nrow(data))
}

hi_pos<- which(Data$Type == "Heat Index")
Data$ID[hi_pos]<- sapply(Data$ID[hi_pos], function(s){strsplit(strsplit(s, ",")[[1]][1], "\\(")[[1]][2]})

Data$Type<- as.factor(Data$Type)
levels(Data$Type)<- c(expression(paste("Quant. ", "Heat ", "Index")), expression(paste("Baseline ", "(", lambda, ")")), expression(paste("Effectiveness ", "(", tau, ")")))
Data$Region<- factor(Data$Region, levels=Regions)

Data<- Data[which(Data$ID %in% counties),] # removing QHI from other counties

plot_df<- data.frame(Type=rep(Data$Type, each=n_days),
                     County=rep(Data$ID, each=n_days),
                     Region=rep(Data$Region, each=n_days),
                     Value=as.vector(t(as.matrix(Data[,1:n_days]))),
                     Day=rep(seq(1,153),nrow(Data)))


p<- ggplot(plot_df[sample(1:nrow(plot_df), 10000),], aes(x=Day, y=Value, color=Region, fill=Region,
                        group=County)) + 
  stat_smooth(method="loess", span=0.5)

p + facet_grid(cols = vars(Type), scales = "free_y", labeller = label_parsed)


p2<- ggplot(plot_df, aes(x=Day, y=Value, color=Region, fill=Region)) + 
  geom_smooth()

p2 + facet_grid(rows = vars(Type), scales = "free_y", labeller = label_parsed) # the plot included in the paper!


p3<- ggplot(plot_df, aes(x=Day, y=Value, color=Region, fill=Region)) +
  geom_smooth(stat = "summary", fun.args = list(conf.int = 0.95))

p3 + facet_grid(rows = vars(Type), scales = "free_y", labeller = label_parsed)


p4<- ggplot(plot_df, aes(x=Day, y=Value, color=Region, fill=Region)) +
  stat_summary(geom="ribbon", alpha = 0.4, lty="blank",
               fun.min = function(x)quantile(x, 0.25),
               fun.max = function(x)quantile(x, 0.75)
               ) +
  geom_smooth(stat="summary", fun = median)

p4 + facet_grid(rows = vars(Type), scales = "free_y", labeller = label_parsed)

