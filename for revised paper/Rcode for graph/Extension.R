data1<- read.csv("/Users/yicheng/Dropbox/COVID-19-Mask/RED/Second_Review/data/All_graphs/data/theta2.csv",head=T)
data2<- read.csv("/Users/yicheng/Dropbox/COVID-19-Mask/RED/Second_Review/data/All_graphs/data/hetero.csv",head=T)
data3<- read.csv("/Users/yicheng/Dropbox/COVID-19-Mask/RED/Second_Review/data/All_graphs/data/hetero_old0.27.csv",head=T)

data11<- data.frame(m0=c(data1$m0,data1$m0,data1$m0,data1$m0),fatalities=c(data1$Optimal.fatalities, data1$GRBT.fatalities, data1$SRA1.fatalities, data1$SRA2.fatalities),name=c(rep("Optimal",71), rep("GRBT",71), rep("SRA-I",71), rep("SRA-II",71)))
data12<- data.frame(m0=c(data2$m0,data2$m0,data2$m0,data2$m0),fatalities=c(data2$Optimal.fatalities, data2$GRBT.fatalities, data2$SRA1.fatalities, data2$SRA2.fatalities),name=c(rep("Optimal",71), rep("GRBT",71), rep("SRA-I",71), rep("SRA-II",71)))
data13<- data.frame(m0=c(data3$m0,data3$m0,data3$m0,data3$m0),fatalities=c(data3$Optimal.fatalities, data3$GRBT.fatalities, data3$SRA1.fatalities, data3$SRA2.fatalities),name=c(rep("Optimal",71), rep("GRBT",71), rep("SRA-I",71), rep("SRA-II",71)))

data21<- data.frame(m0=c(data1$m0,data1$m0),ic=c(data1$IC.without.masks,data1$IC.with.masks),name=c(rep("Agents without masks",71), rep("Agents with masks",71)))
data22<- data.frame(m0=c(data2$m0,data2$m0),ic=c(data2$IC.without.masks,data2$IC.with.masks),name=c(rep("Agents without masks",71), rep("Agents with masks",71)))
data23<- data.frame(m0=c(data3$m0,data3$m0),ic=c(data3$IC.without.masks,data3$IC.with.masks),name=c(rep("Agents without masks",71), rep("Agents with masks",71)))

ndata12<- data.frame(m0=c(data11$m0,data12$m0,data13$m0),fatalities=c((1/10)*data11$fatalities,data12$fatalities,data13$fatalities),name=c(rep("Optimal",71),rep("GRBT",71),rep("SRA-I",71),rep("SRA-II",71),rep("Optimal",71),rep("GRBT",71),rep("SRA-I",71),rep("SRA-II",71),rep("Optimal",71),rep("GRBT",71),rep("SRA-I",71),rep("SRA-II",71)),case=c(rep("A: 20% of infectious agents are identified",284),rep("B: Heterogeneous agents",284),rep("C: Heterogeneous agents with lower mask protection",284)))

ndata22<- data.frame(m0=c(data21$m0,data22$m0,data23$m0),ic=c(data21$ic,data22$ic,data23$ic),name=c(rep("Agents without masks",71), rep("Agents with masks",71),rep("Agents without masks",71), rep("Agents with masks",71),rep("Agents without masks",71), rep("Agents with masks",71)),case=c(rep("A: 20% of infectious agents are identified",142),rep("B: Heterogeneous agents",142), rep("C: Heterogeneous agents with lower mask protection",142)))

library(lattice)
library(latticeExtra)

ndata12$name<- factor(ndata12$name, c("SRA-I","SRA-II","GRBT","Optimal"))
key.variety<-
list(space = "top", text = list(levels(ndata12$name)),columns = 4,
points = list(pch = c(19,1,19,1),col = c("slategray4","slategray4","gray20","firebrick2")))

tp1<- xyplot(fatalities*100000~m0|case, groups = name, data=ndata12, grid=F,layout=c(1,3),col=c("slategray4","slategray3","gray20","firebrick2"),
index.cond=list(c(3,2,1)),#this provides the order of the panels
key = key.variety,pch=c(16,1,16,1),cex=0.75,
xlab="Coverage rate of new masks to the population at t=1", ylab="Mortality (A: 10,000 population, B: 100,000 population, C: 100,000 population)",type="p",ylim=c(0.75*100000*min(ndata12$fatalities),1.25*100000*max(ndata12$fatalities)),xlim=c(0.08,0.82),
par.settings = theEconomist.theme(box = "transparent",with.bg =T),
lattice.options = theEconomist.opts())

ndata22$name<- factor(ndata22$name, c("Agents with masks","Agents without masks"))

key.variety<-
list(space = "top", text = list(levels(ndata22$name)),columns = 2,
lines = list(lwd=c(2,2),lty = c(1,1),col = c("gray20","firebrick2")))

tp2<- xyplot(ic~m0|case, groups = name, data=ndata22, grid=F,layout=c(1,3),col=c("gray20","firebrick2"),
index.cond=list(c(3,2,1)),#this provides the order of the panels
key = key.variety,lty = c(1,1),cex=0.5,
xlab="Coverage rate of new masks to the population at t=1", ylab="Gains from early participation in the GRBT",type="l",ylim=c(1.75*min(ndata22$ic),1.75*max(ndata22$ic)),xlim=c(0.08,0.82),
par.settings = theEconomist.theme(box = "transparent",with.bg =T),
lattice.options = theEconomist.opts())

m<- 3.75
pdf("/Users/yicheng/Dropbox/COVID-19-Mask/RED/Second_Review/data/All_graphs/graph/Extension.pdf",width =1.618*1.85*m, height = 2.5*m) 
plot(tp1, split = c(1, 1, 2, 1))
plot(tp2, split = c(2, 1, 2, 1), newpage = FALSE)
dev.off( )
