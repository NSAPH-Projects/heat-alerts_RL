library(lubridate)
library(ggplot2)
library(dplyr)

load("data/HARL_prelim_image.RData")

#### Detailed summaries of cumulative heat alerts:

summer$day<- day(summer$Date)
last_day<- summer[which(summer$month == 9 & summer$day == 30),]
# x1<- last_day$alert_sum
# hist(x1, 
#      labels = paste0(round(hist(x1, plot = FALSE)$counts / length(x1) * 100, 1), "%"))

last_day %>% 
  count(AS = 2*floor(alert_sum/2)) %>% 
  mutate(pct = round(prop.table(n),2)) %>% 
  ggplot(aes(x = AS, y = pct, 
                     label = scales::percent(pct))) + 
  geom_col(position = 'dodge') + 
  geom_text(position = position_dodge(width = .9),    # move to center of bars
            vjust = -0.5,    # nudge above top of bar
            size = 3) + 
  scale_y_continuous(labels = scales::percent) + 
  ggtitle("Cumulative Alerts Per County-Summer")

quantile(last_day$alert_sum, seq(0.5,1,0.05))
quantile(last_day$alert_sum, seq(0,0.5,0.05))

#### Detailed summaries of deaths:



#### For STAT 234 report:

stats<- round(sapply(summer[,c("HImaxF_PopW", "HI_3days",
                               "alert_sum", "Pop_density", 
                               "Med.HH.Income", "N")], summary),2)
# for(i in 1:ncol(stats)){
#   print(paste(round(stats[,i]), collapse = ","))
# }

for(i in 1:nrow(stats)){
  print(paste(round(stats[i,]), collapse = " & "))
}

