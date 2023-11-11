
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

# Data$Type[hi_pos]<- "Quant. Heat Index"
# Data$Type[which(Data$Type == "Lambda")]<- as.expression(bquote(~ "Baseline ("* lambda *")"))
# Data$Type[which(Data$Type == "Tau")]<- as.expression(bquote(~ "Effectiveness ("* tau *")"))
Data$Type<- as.factor(Data$Type)
levels(Data$Type)<- c(expression(paste("Quant. ", "Heat ", "Index")), expression(paste("Baseline ", "(", lambda, ")")), expression(paste("Effectiveness ", "(", tau, ")")))
Data$Region<- factor(Data$Region, levels=Regions)

# subset<- sample(1:nrow(Data), 1000)
# Data<- Data[subset,]
Data<- Data[which(Data$ID %in% counties),] # removing QHI from other counties

plot_df<- data.frame(Type=rep(Data$Type, each=n_days),
                     County=rep(Data$ID, each=n_days),
                     Region=rep(Data$Region, each=n_days),
                     Value=as.vector(t(as.matrix(Data[,1:n_days]))),
                     Day=rep(seq(1,153),nrow(Data)))

# p<- ggplot(plot_df, aes(x=Day, y=Value, color=Region, fill=Region,
#                         group=County)) + 
#   stat_smooth(method="loess", span=0.5)
# 
# p + facet_grid(rows = vars(Type), scales = "free_y", labeller = label_parsed)

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
# # changing color of ribbon: https://gist.github.com/valentinitnelav/becf4704c0eef0180546f958e453fc1e
# scale_fill_manual(name = "region",
#                   breaks = c("Cold", "Hot Dry", "Hot Humid", "Mixed Humid", "Marine"),
#                   values = c("Cold"="#F8766D",
#                              "Hot Dry"="#A3A500",
#                              "Hot Humid"="#00BF7D",
#                              "Mixed Humid"="#00B0F6",
#                              "Marine"="#E76BF3"
#                              )) # found hex codes at https://www.statology.org/ggplot-default-colors/

p4 + facet_grid(rows = vars(Type), scales = "free_y", labeller = label_parsed)


################ OLD:

for(r in Region){
  data<- read.csv(paste0("heat_alerts/time_series_data/HI-lam-tau_", r, ".csv"))[,-1]
  data[which(data$Type == "Tau"),1:n_days]<- data[which(data$Type == "Tau"),1:n_days]
  data[which(data$Type == "Tau"), "Type"]<- "Tau"
  
  hi_data<- data[which(data$Type == "Heat Index"),]
  hi_data$Type<- "HI Quantile"
  # hi_data$Year<- sapply(hi_data$ID, function(s){strsplit(strsplit(s, " ")[[1]][2], ")")[[1]][1]})
  hi_data$ID<- sapply(hi_data$ID, function(s){strsplit(strsplit(s, ",")[[1]][1], "\\(")[[1]][2]})
  # HI_data<- aggregate(. ~ Type + ID, hi_data[,-ncol(hi_data)], mean)
  # 
  # Data<- aggregate(. ~ Type + ID, 
  #                  data[which(data$Type != "Heat Index"),], mean)
  # Data<- rbind(Data, HI_data)
  
  Data<- rbind(hi_data, data[which(data$Type != "Heat Index"),])
  
  plot_df<- data.frame(Type=rep(Data$Type, each=n_days),
                       ID=rep(Data$ID, each=n_days),
                       Value=as.vector(t(as.matrix(Data[,1:n_days]))),
                       Day=rep(seq(1,153),nrow(Data)))
  
  # p<- ggplot(plot_df, aes(x=Day, y=Value, color=Type)) +
  #   geom_point(alpha=0.05) +
  #   # ylim(0, 1.75) + 
  #   geom_smooth() +
  #   # geom_line() +
  #   ggtitle(paste("Climate Region:", r))
  # 
  # print(p)
  
  region_counties<- unique(data[which(data$Type != "Heat Index"),"ID"])

  for(k in region_counties){
    p<- ggplot(plot_df[which(plot_df$ID == k),], aes(x=Day, y=Value, color=Type)) +
      geom_point(alpha=0.2) +
      geom_smooth() +
      # geom_line() +
      ggtitle(paste0("Climate Region: ", r, ", County: ", k))
    print(p)
  }
  
  print(r)
}
