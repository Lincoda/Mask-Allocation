data<- read.csv("/Users/yicheng/Dropbox/COVID-19-Mask/RED/Second_Review/data/All_graphs/data/230504 EndoU benchmark, graph.csv",head=T)
attach(data)

n=71
#the number of simulations
data1<- data.frame(m0=c(m,m,m,m),fatalities=c(Optimal.fatalities, GRBT.fatalities,SRA2.fatalities,SRA1.fatalities),name=c(rep("Optimal",n), rep("GRBT",n), rep("SRA-II",n), rep("SRA-I",n)))
data2<- data.frame(m0=c(m,m),ic=c(ICphi,ICn),name=c(rep("Agents without masks",n), rep("Agents with masks",n)))
data3<- data.frame(m0=c(m,m),ip=c(P2phi,P3phi),name=c(rep("Without masks at t=1",n), rep("Without masks at t=2",n)))

library(lattice)
library(latticeExtra)

data3$name<- factor(data3$name, c("Without masks at t=1","Without masks at t=2"))
key.variety<-
  list(space = "top", text = list(levels(data3$name)),columns = 2,cex=1,
       lines = list(lwd=c(2,2),lty = c(1,1),col = c("black","firebrick2")))

tp5<- xyplot(ip~m0, groups = name, data=data3, grid=F,layout=c(1,1),col=c("black","firebrick2"),
             key = key.variety,lty = c(1,1),cex=0.5,
             xlab="Coverage rate of new masks to the population at t=1", ylab="Infection probabilities in the GRBT",type="l",ylim=c(0.1*min(data3$ip),1.1*max(data3$ip)),xlim=c(0.08,0.82),
             par.settings = theEconomist.theme(box = "transparent",with.bg =T),
             lattice.options = theEconomist.opts())

data2$name<- factor(data2$name, c("Agents with masks","Agents without masks"))
key.variety<-
  list(space = "top", text = list(levels(data2$name)),columns = 2,cex=1,
       lines = list(lwd=c(2,2),lty = c(1,1),col = c("black","firebrick2")))

tp4<- xyplot(ic~m0, groups = name, data=data2, grid=F,layout=c(1,1),col=c("black","firebrick2"),
             key = key.variety,lty = c(1,1),cex=0.5,
             xlab="Coverage rate of new masks to the population at t=1", ylab="Gains from early participation in the GRBT",type="l",ylim=c(1.75*min(data2$ic),1.75*max(data2$ic)),xlim=c(0.08,0.82),
             par.settings = theEconomist.theme(box = "transparent",with.bg =T),
             lattice.options = theEconomist.opts())

data1$name<- factor(data1$name, c("SRA-I","SRA-II","GRBT","Optimal"))
key.variety<-
list(space = "top", text = list(levels(data1$name)),columns = 4,cex=1,
points = list(pch = c(19,1,19,1),col = c("slategray4","slategray4","black","firebrick2")))

tp3<- xyplot(fatalities*100000~m0, groups = name, data=data1, grid=F,layout=c(1,1),col=c("slategray4","slategray3","black","firebrick2"),
key = key.variety,pch=c(16,1,16,1),cex=0.75,
xlab="Coverage rate of new masks to the population at t=1", ylab="Mortality (100,000 population)",type="p",ylim=c(0.75*100000*min(data1$fatalities),1.05*100000*max(data1$fatalities)),xlim=c(0.08,0.82),
par.settings = theEconomist.theme(box = "transparent",with.bg =T),
lattice.options = theEconomist.opts())

m<- 3.75
pdf("/Users/yicheng/Dropbox/COVID-19-Mask/RED/Second_Review/data/All_graphs/graph/benchmark_EU.pdf",width =1.618*1.8*m, height = 2*m) 
print(tp3, split=c(1,1,2,2), more=TRUE)
print(tp4, position = c(0.5,0.5,0.985,1), more=TRUE)
print(tp5, position = c(0.25,0,0.75,0.5), more=FALSE)
dev.off( )
