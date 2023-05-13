data<- read.csv("/Users/yicheng/Dropbox/COVID-19-Mask/要上傳的程式碼/R graphs/gamma09.csv",head=T)
attach(data)

data1<- data.frame(m0=c(m0,m0,m0,m0),fatalities=c(Optimal.fatalities, GRBT.fatalities,SRA2.fatalities,SRA1.fatalities),name=c(rep("Optimal",71), rep("GRBT",71), rep("SRA-II",71), rep("SRA-I",71)))
data2<- data.frame(m0=c(m0,m0),ic=c(IC.without.masks,IC.with.masks),name=c(rep("Agents without masks",71), rep("Agents with masks",71)))

library(lattice)
library(latticeExtra)

key.variety<-
list(space = "top", text = list(levels(data2$name)),columns = 2,cex=1,
lines = list(lty = c(1,1),col = c("gray20","firebrick2")))

tp4<- xyplot(ic~m0, groups = name, data=data2, grid=F,layout=c(1,1),col=c("gray20","firebrick2"),
key = key.variety,lty = c(1,1),cex=0.5,
xlab="Coverage rate of new masks to the population at t=1", ylab="Gains from early participation in the GRBT",type="l",ylim=c(1.75*min(data2$ic),1.75*max(data2$ic)),xlim=c(0.08,0.82),
par.settings = theEconomist.theme(box = "transparent",with.bg =T),
lattice.options = theEconomist.opts())

data1$name<- factor(data1$name, c("SRA-I","SRA-II","GRBT","Optimal"))
key.variety<-
list(space = "top", text = list(levels(data1$name)),columns = 4,cex=1,
points = list(pch = c(19,1,19,1),col = c("slategray4","slategray4","gray20","firebrick2")))

tp3<- xyplot(fatalities*100000~m0, groups = name, data=data1, grid=F,layout=c(1,1),col=c("slategray4","slategray3","gray20","firebrick2"),
key = key.variety,pch=c(16,1,16,1),cex=0.75,
xlab="Coverage rate of new masks to the population at t=1", ylab="Mortality (100,000 population)",type="p",ylim=c(0.75*100000*min(data1$fatalities),1.05*100000*max(data1$fatalities)),xlim=c(0.08,0.82),
par.settings = theEconomist.theme(box = "transparent",with.bg =T),
lattice.options = theEconomist.opts())

m<- 3.75
pdf("/Users/yicheng/Dropbox/COVID-19-Mask/要上傳的程式碼/R graphs/final_graphs/Robust_gamma.pdf",width =1.618*1.8*m, height = 1*m) 
plot(tp3, split = c(1, 1, 2, 1))
plot(tp4, split = c(2, 1, 2, 1), newpage = FALSE)
dev.off( )


