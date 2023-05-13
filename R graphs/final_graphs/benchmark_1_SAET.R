data<- read.csv("/Users/yicheng/Dropbox/COVID-19-Mask/要上傳的程式碼/R graphs/benchmark.csv",head=T)
attach(data)

data1<- data.frame(m0=c(m0,m0,m0),fatalities=c(Optimal.fatalities, GRBT.fatalities,SRA1.fatalities),name=c(rep("Optimal",71), rep("GRBT",71), rep("SRA-I",71)))

library(lattice)
library(latticeExtra)

data1$name<- factor(data1$name, c("SRA-I","GRBT","Optimal"))
key.variety<-
list(space = "top", text = list(levels(data1$name)),columns = 3,cex=1,
points = list(pch = c(19,19,1),col = c("slategray4","black","firebrick2")))

tp1<- xyplot(fatalities*100000~m0, groups = name, data=data1, grid=F,layout=c(1,1),col=c("slategray4","black","firebrick2"),
key = key.variety,pch=c(16,16,1),cex=0.75,
xlab="Coverage rate of new masks to the population at t=1", ylab="Mortality (100,000 population)",type="p",ylim=c(0.75*100000*min(data1$fatalities),1.05*100000*max(data1$fatalities)),xlim=c(0.08,0.82),
par.settings = theEconomist.theme(box = "transparent",with.bg =T),
lattice.options = theEconomist.opts())

m<- 3.75
pdf("/Users/yicheng/Dropbox/COVID-19-Mask/要上傳的程式碼/R graphs/final_graphs/benchmark_1_SAET.pdf",width =1.618*m, height = 1*m) 
tp1
dev.off( )
