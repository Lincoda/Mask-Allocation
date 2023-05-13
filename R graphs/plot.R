data<- read.csv("D:\\R\\improve accuracy\\hetero_old0.27.csv",head=T)
attach(data)
data2<- data.frame(m0=c(m0,m0),ic=c(IC.without.masks,IC.with.masks),name=c(rep("Agents without masks",71), rep("Agents with masks",71)))
data3<- data.frame(m0=c(m0,m0,m0,m0),fatalities=c(Optimal.fatalities, GRBT.fatalities,SRA1.fatalities,SRA2.fatalities),name=c(rep("Optimal",71), rep("GRBT",71), rep("SRA1",71), rep("SRA2",71)))

library(lattice)
library(latticeExtra)

data2$name<- factor(data2$name, c("Agents with masks","Agents without masks"))

key.variety<-
  list(space = "top", text = list(levels(data2$name)),columns = 2,
       lines = list(lty = c(1,1),col = c("black","firebrick2")))

tp4<- xyplot(ic~m0, groups = name, data=data2, grid=F,layout=c(1,1),col=c("black","firebrick2"),
             key = key.variety,lty = c(1,1),cex=1,
             xlab="Initial mask coverage rate", ylab="Gains from early participation in the GRBT",type="l",ylim=c(-0.6,1),xlim=c(0.08,0.82),
             par.settings = theEconomist.theme(box = "transparent",with.bg =T),
             lattice.options = theEconomist.opts())


data3$name<- factor(data3$name, c("SRA1","SRA2","GRBT","Optimal"))
key.variety<-
  list(space = "top", text = list(levels(data3$name)),columns = 4,
       lines = list(lty = c(1,1,1,1),col = c("slategray","blue","black","firebrick2")))

tp3<- xyplot(fatalities*100000~m0, groups = name, data=data3, grid=F,layout=c(1,1),col=c("slategray","blue","black","firebrick2"),
             key = key.variety,lty=c(1,1,1,1),cex=1,
             xlab="Initial mask coverage rate", ylab="Mortality (100000 population)",type="l",ylim=c(16,24),xlim=c(0.08,0.82),
             par.settings = theEconomist.theme(box = "transparent",with.bg =T),
             lattice.options = theEconomist.opts())

plot(tp3, split = c(1, 1, 2, 1))
plot(tp4, split = c(2, 1, 2, 1), newpage = FALSE)
