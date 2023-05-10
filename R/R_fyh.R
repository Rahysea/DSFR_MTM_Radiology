#----------
# Regression
#----------

library(data.table)
library(ResourceSelection)
dd = data.frame(fread("datainfo.CSV", header = TRUE))[,-2]
dd.train = dd[which(dd$dataset == 1),]
dd.internal_test = dd[which(dd$dataset == 2),]
dd.external_test = dd[which(dd$dataset == 0),]

listVars = colnames(dd)[-c(1:2)]
para = listVars[-26]

index_p = c();name = NULL
res = NULL
for(i in 1:length(para)){
  formulA = as.formula(paste("MTM~",para[i],sep=""))
  fit = glm(formulA, family = binomial(link=logit), data = dd.train)
  co = summary(fit)
  if(sum(as.numeric(co$coefficients[-1,4]<0.05))>0){index_p = c(index_p,i)}
  res = rbind(res,co$coefficients[-1,])
  name = c(name,rownames(co$coefficients)[-1])
  print(i)
}

res = data.frame(res)
colnames(res) = c("BETA","SE","Z","p")
rownames(res) = name
res$Low = exp(res$BETA-1.96*res$SE); res$Up= exp(res$BETA+1.96*res$SE);
res$OR = exp(res$BETA)
res$esitmate = paste(round(res$OR,3)," (",round(res$Low,3),", ",round(res$Up,3),")",sep="")
# write.csv(res,"res_logistic_uni.csv",row.names=T)



para.all = para[index_p]
formulA = as.formula(paste("MTM~",paste(para.all, collapse = '+'),sep=""))
fit.multi = glm(formulA, family = binomial(link=logit), data = dd.train)
co.multi = summary(fit.multi)
res.multi = data.frame(co.multi$coefficients)
colnames(res.multi) = c("BETA","SE","Z","p")
res.multi$Low = exp(res.multi$BETA-1.96*res.multi$SE); res.multi$Up= exp(res.multi$BETA+1.96*res.multi$SE);
res.multi$OR = exp(res.multi$BETA)
res.multi$esitmate = paste(round(res.multi$OR,3)," (",round(res.multi$Low,3),", ",round(res.multi$Up,3),")",sep="")
# write.csv(res.multi,"res_logistic_multi.csv",row.names=T)


fit.step.cat = step(fit.multi.cat)
co.multi = summary(fit.step.cat)
res.multi = data.frame(co.multi$coefficients)
colnames(res.multi) = c("BETA","SE","Z","p")
res.multi$Low = exp(res.multi$BETA-1.96*res.multi$SE); res.multi$Up= exp(res.multi$BETA+1.96*res.multi$SE);
res.multi$OR = exp(res.multi$BETA)
res.multi$esitmate = paste(round(res.multi$OR,3)," (",round(res.multi$Low,3),", ",round(res.multi$Up,3),")",sep="")
#write.csv(res.multi,"res_logistic_multi_cat_step.csv",row.names=T)


fit.step = step(fit.multi)
co.multi = summary(fit.step)
res.multi = data.frame(co.multi$coefficients)
colnames(res.multi) = c("BETA","SE","Z","p")
res.multi$Low = exp(res.multi$BETA-1.96*res.multi$SE); res.multi$Up= exp(res.multi$BETA+1.96*res.multi$SE);
res.multi$OR = exp(res.multi$BETA)
res.multi$esitmate = paste(round(res.multi$OR,3)," (",round(res.multi$Low,3),", ",round(res.multi$Up,3),")",sep="")

#----------
# Nomogram
#----------
para.DLR = names(fit.step$coefficients)[-c(1)]
para.CR = names(fit.step.cat$coefficients)[-c(1)]

library(rms)
data.train = dd.train[,-1]
data.train$AFP <- factor(data.train$AFP, labels = c("≤ 100 ng/ml", "> 100 ng/ml"))
para3 <- gsub(".", " ", para.DLR, fixed = TRUE)

colnames(data.train) <- gsub(".", " ", colnames(data.train), fixed = TRUE)
formula.3 <- as.formula(paste('MTM~',paste(para.DLR, collapse = '+')))
formula.2 <- as.formula(paste("MTM ~", paste(paste0("`", para3, "`"), collapse = " + ")))
ddist <- datadist(data.train); options(datadist='ddist')
f <- lrm(formula.2,data=data.train)
nom <- nomogram(f, fun=function(x)1/(1+exp(-x)),  # or fun=plogis
                fun.at=c(.001,.01,seq(.1,.9,by=.3),.95,.99,.999),
                funlabel="Probability of MTM")

tiff("nomogram.tiff",width = 3500, height = 2000, res = 300)
attr(nom, "info")$lp <- FALSE

cex_axis_size <- 1.3 
cex_lab_size <- 0.5

axis_label_weight <- 2 
axis_tick_weight <- 1

bold_font <- 2

plot(nom, vnames = "labels", label.dist = 0.3, family = "Times New Roman", 
     cex.lab = cex_lab_size, cex.axis = cex_axis_size,
     lwd.lab = axis_label_weight, lwd.axis = axis_tick_weight,
     font.axis = bold_font, font.lab = bold_font)
dev.off()

#----------
# ROC
#----------	
formul.CR = as.formula(paste("MTM~",paste(para.CR, collapse = '+'),sep=""))
fit.multi.CR = glm(formul.CR, binomial(logit), data = dd.train)
co.multi = summary(fit.multi.CR)
res.multi = data.frame(co.multi$coefficients)
colnames(res.multi) = c("BETA","SE","Z","p")
res.multi$Low = exp(res.multi$BETA-1.96*res.multi$SE); res.multi$Up= exp(res.multi$BETA+1.96*res.multi$SE);
res.multi$OR = exp(res.multi$BETA)
res.multi$esitmate = paste(round(res.multi$OR,3)," (",round(res.multi$Low,3),", ",round(res.multi$Up,3),")",sep="")
# write.csv(res.multi,"res_CR1.csv",row.names=T)

formul.DLR = as.formula(paste("MTM~",paste(para.DLR, collapse = '+'),sep=""))
fit.multi.DLR = glm(formul.DLR, binomial(logit), data = dd.train)
co.multi = summary(fit.multi.DLR)
res.multi = data.frame(co.multi$coefficients)
colnames(res.multi) = c("BETA","SE","Z","p")
res.multi$Low = exp(res.multi$BETA-1.96*res.multi$SE); res.multi$Up= exp(res.multi$BETA+1.96*res.multi$SE);
res.multi$OR = exp(res.multi$BETA)
res.multi$esitmate = paste(round(res.multi$OR,3)," (",round(res.multi$Low,3),", ",round(res.multi$Up,3),")",sep="")
# write.csv(res.multi,"res_DLR.csv",row.names=T)


library(pROC)
dd$lp.DLR = predict(fit.multi.DLR, newdata = dd,type = "response")
dd$lp.CR = predict(fit.multi.CR, newdata = dd,type = "response")
# write.csv(dd, "dd.linearpredictor.csv", row.names = TRUE)

dd.train$lp.DLR = predict(fit.multi.DLR, newdata = dd.train,type = "response")
dd.internal_test $lp.DLR = predict(fit.multi.DLR, newdata = dd.internal_test,type = "response")
dd.external_test$lp.DLR = predict(fit.multi.DLR, newdata = dd.external_test,type = "response")

dd.train$lp.CR = predict(fit.multi.CR, newdata = dd.train ,type = "response")
dd.internal_test$lp.CR = predict(fit.multi.CR, newdata = dd.internal_test,type = "response")
dd.external_test$lp.CR = predict(fit.multi.CR, newdata = dd.external_test,type = "response")

#	DL Radiomics-based model
auc.1 = roc(dd.train$MTM,	dd.train$lp.DLR, ci = TRUE) 
auc.2 = roc(dd.internal_test$MTM,	dd.internal_test$lp.DLR, ci = TRUE) 
auc.3 = roc(dd.external_test$MTM,	dd.external_test$lp.DLR, ci = TRUE) 

#	Clinical-radiological model
auc.10 = roc(dd.train$MTM,	dd.train$lp.CR, ci = TRUE) 
auc.11 = roc(dd.internal_test$MTM,	dd.internal_test$lp.CR, ci = TRUE)
auc.12 = roc(dd.external_test$MTM,	dd.external_test$lp.CR, ci = TRUE)

roc.P.3 = roc.test(auc.1, auc.10)
roc.P.6 = roc.test(auc.2, auc.11)
roc.P.9 = roc.test(auc.3, auc.12)

auc.1; auc.10

res = NULL
res.1 = coords(auc.1, "best", ret = "all", best.method=c("youden"), transpose = FALSE)
res.10 = coords(auc.10, "best", ret = "all", best.method=c("youden"), transpose = FALSE)

ress.1 = ci.coords(auc.1, "best", ret = "all", best.method=c("youden"),best.policy = c("random"), transpose = FALSE)
ress.10 = ci.coords(auc.10, "best", ret = "all", best.method=c("youden"),best.policy = c("random"), transpose = FALSE)

res = rbind(res.1,res.10)
AUC = c(auc.1$auc,auc.10$auc)
AUC.low = c(auc.1$ci[1], auc.10$ci[1])
AUC.up = c(auc.1$ci[3], auc.10$ci[3])
RES = res[,c("threshold","specificity","sensitivity","accuracy","npv","ppv")]

ress = rbind(c(ress.1$threshold[c(1,3)], ress.1$specificity[c(1,3)], ress.1$sensitivity[c(1,3)], ress.1$accuracy[c(1,3)], ress.1$npv[c(1,3)], ress.1$ppv[c(1,3)]),
             c(ress.10$threshold[c(1,3)], ress.10$specificity[c(1,3)], ress.10$sensitivity[c(1,3)], ress.10$accuracy[c(1,3)], ress.10$npv[c(1,3)], ress.10$ppv[c(1,3)]))


RES = data.frame(Models = c("Training DL radiomics-based model", "Training Clinical-radiological model"), AUC, AUC.low, AUC.up, RES, ress)
colnames(RES) = c("Models","AUC","AUC.low","AUC.up","threshold","specificity","sensitivity","accuracy","npv","ppv",
                  "threshold.low","threshold.up","specificity.low","specificity.up","sensitivity.low","sensitivity.up","accuracy.low","accuracy.up","npv.low","npv.up","ppv.low","ppv.up")
row.names(RES) = NULL
# write.csv(RES, "res_diagnosis_training.csv", row.names=FALSE, quote=F)



library(caret)
dd.train$lp.DLR.cat <- factor(ifelse(predict(fit.multi.DLR, dd.train, type="response")>0.234338736,"1","0"),levels=c(1,0))
dd.internal_test$lp.DLR.cat <- factor(ifelse(predict(fit.multi.DLR, dd.internal_test, type="response")>0.234338736,"1","0"),levels=c(1,0))
dd.external_test$lp.DLR.cat <- factor(ifelse(predict(fit.multi.DLR, dd.external_test, type="response")>0.234338736,"1","0"),levels=c(1,0))
dd.train$lp.CR.cat <- factor(ifelse(predict(fit.multi.CR, dd.train, type="response")>0.15260921,"1","0"),levels=c(1,0))
dd.internal_test$lp.CR.cat <- factor(ifelse(predict(fit.multi.CR, dd.internal_test, type="response")>0.15260921,"1","0"),levels=c(1,0))
dd.external_test$lp.CR.cat <- factor(ifelse(predict(fit.multi.CR, dd.external_test, type="response")>0.15260921,"1","0"),levels=c(1,0))

res.1 = confusionMatrix(dd.train$lp.DLR.cat, factor(dd.train$MTM,levels=c(1,0)))
res.2 = confusionMatrix(dd.train$lp.CR.cat, factor(dd.train$MTM,levels=c(1,0)))
res.3 = confusionMatrix(dd.internal_test$lp.DLR.cat, factor(dd.internal_test$MTM,levels=c(1,0)))
res.4 = confusionMatrix(dd.internal_test$lp.CR.cat, factor(dd.internal_test$MTM,levels=c(1,0)))
res.5 = confusionMatrix(dd.external_test$lp.DLR.cat, factor(dd.external_test$MTM,levels=c(1,0)))
res.6 = confusionMatrix(dd.external_test$lp.CR.cat, factor(dd.external_test$MTM,levels=c(1,0)))
#----------
library(epiR)	
ress.1 = epi.tests(res.1$table, conf.level = 0.95)
ress.2 = epi.tests(res.2$table, conf.level = 0.95)
ress.3 = epi.tests(res.3$table, conf.level = 0.95)
ress.4 = epi.tests(res.4$table, conf.level = 0.95)
ress.5 = epi.tests(res.5$table, conf.level = 0.95)
ress.6 = epi.tests(res.6$table, conf.level = 0.95)
RES = rbind(data.frame(Group = "train.DLR",rbind(c("accuracy",res.1$overall[c(1,3,4)]),ress.1$detail)),
            data.frame(Group = "train.CR",rbind(c("accuracy",res.2$overall[c(1,3,4)]),ress.2$detail)),
            data.frame(Group = "internal_test.DLR",rbind(c("accuracy",res.3$overall[c(1,3,4)]),ress.3$detail)),
            data.frame(Group = "internal_test.CR",rbind(c("accuracy",res.4$overall[c(1,3,4)]),ress.4$detail)),
            data.frame(Group = "external_test.DLR",rbind(c("accuracy",res.5$overall[c(1,3,4)]),ress.5$detail)),
            data.frame(Group = "external_test.CR",rbind(c("accuracy",res.6$overall[c(1,3,4)]),ress.6$detail)))

# write.csv(RES,"result.internal_test.external_test.accuracy.csv")

#----------
# ROC plot
#----------
library(rmda)
library(pROC)

pdf("Figure ROC.pdf", width = 18, height = 6)
par(mfrow=c(1,3),cex.axis=1.6,cex.lab=1.6,cex.main=1.6,lwd=0.1,
    mgp=c(2,0.5,0),tcl=-0.2,font.axis=1.1, text.font=2,
    font.lab=1.1,mar=c(5.1,4.1,2.5,2.1))
plot.roc(auc.1, col="#FF0000", percent = FALSE,
         print.auc.x=0.5, print.auc.y=0.4)
plot(auc.10, add=TRUE, col="#009933",
     print.auc.x=0.5, print.auc.y=0.4)
legend("bottomright", legend=c(	
  paste("DL radiomics-based nomogram: AUC=0.91",sep=""),
  paste("Clinical-radiological model: AUC=0.77",sep="")), 
  col=c("#FF0000","#009933"), lwd=2, cex=1.6, text.font=2)

plot.roc(auc.2,col="#FF0000",
         print.auc.x=0.5, print.auc.y=0.4)
plot(auc.11,add=TRUE,col="#009933",
     print.auc.x=0.5, print.auc.y=0.4, cex=1.3)
legend("bottomright", legend=c(	
  paste("DL radiomics-based nomogram: AUC=0.87",sep=""),
  paste("Clinical-radiological model: AUC=0.72",sep="")), 
  col=c("#FF0000","#009933"), lwd=2, cex=1.6, text.font=2)

plot.roc(auc.3,col="#FF0000",
         print.auc.x=0.5, print.auc.y=0.4)
plot(auc.12,add=TRUE,col="#009933",
     print.auc.x=0.5, print.auc.y=0.4)
legend("bottomright", legend=c(	
  paste("DL radiomics-based nomogram: AUC=0.89",sep=""),
  paste("Clinical-radiological model: AUC=0.79",sep="")), 
  col=c("#FF0000","#009933"), lwd=2, cex=1.6, text.font=2)
dev.off()

#----------
# calibrate plot
#----------
source("calibrate.my.R")

pdf("Figure calibrate.pdf",width = 15, height = 5)
par(mfrow=c(1, 3),cex.axis=1.8,cex.lab=1.8,cex.main=1.8,lwd=0.1,
    mgp=c(2,0.5,0),tcl=-0.2,font.axis=1.3,
    font.lab=1.1,mar=c(5.1,4.1,2.5,2.1))

f <- lrm(formula.3, data=dd.train,x=TRUE,y=TRUE)
f1 <- lrm(formula.3, data=dd.internal_test,x=TRUE,y=TRUE)
f2 <- lrm(formula.3, data=dd.external_test,x=TRUE,y=TRUE)

cal.train<-calibrate(f, cmethod="hare", method="boot",u=30) 
calibrate.my(cal.train,xlab = "Nomogram-predicted probility",subtitles=F)
# mtext(expression(paste("(A)",sep="")),outer=F,at=0.01,cex=1.0)

cal.internal_test<-calibrate(f1, cmethod="hare", method="boot",u=13) 
calibrate.my(cal.internal_test,xlab = "Nomogram-predicted probility",subtitles=F)
# mtext(expression(paste("(B)",sep="")),outer=F,at=0.01,cex=1.0)

cal.external_test<-calibrate(f2, cmethod="hare", method="boot",u=30) 
calibrate.my(cal.external_test,xlab = "Nomogram-predicted probility",subtitles=F)
# mtext(expression(paste("(C)",sep="")),outer=F,at=0.01,cex=1.0)
dev.off()


#----------
# ROC compare 2
#----------
roc.PP.1 = roc.test(auc.1,auc.2)
roc.PP.2 = roc.test(auc.1,auc.3)
roc.PP.3 = roc.test(auc.2,auc.3)

roc.PP.10 = roc.test(auc.10,auc.11)
roc.PP.11 = roc.test(auc.10,auc.12)
roc.PP.12 = roc.test(auc.11,auc.12)

ROC.res = rbind(c(roc.PP.1$p.value,roc.PP.2$p.value,roc.PP.3$p.value),
                c(roc.PP.10$p.value,roc.PP.11$p.value,roc.PP.12$p.value))
colnames(ROC.res) = c("train-internal_test","train-external_test","internal_test-external_test")
rownames(ROC.res) = c("DL radiomics-based model","clinical-radiological model")
# write.csv(ROC.res, "ROC.compare.csv", row.names = TRUE)

#----------
# DCA plot
#----------
library(survival)
library(pROC)
library(rmda)
library(rms)

formulC = as.formula(paste("MTM~",paste(c(para.DLR,"Obvineco"), collapse = '+'),sep=""))
fit.multi.3 = glm(formulC, data = dd.train)

dd.train$lp.3 = predict(fit.multi.3, newdata = dd.train)
dd.internal_test$lp.3 = predict(fit.multi.3, newdata = dd.internal_test)
dd.external_test$lp.3 = predict(fit.multi.3, newdata = dd.external_test)

pdf("Figure DCA2.pdf",width = 16, height = 6)
par(mfrow=c(1, 3),cex.axis=1.8,cex.lab=1.8,cex.main=1.6,lwd=0.1,
    mgp=c(2,0.5,0),tcl=-0.2,font.axis=1.3,
    font.lab=1.4,mar=c(5.1,4.1,2.5,2.1))

dca.DLR <- decision_curve(MTM~lp.DLR, data = dd.train, bootstraps = 50)
dca.CR <- decision_curve(MTM~lp.CR, data = dd.train, bootstraps = 50)
dca.3 <- decision_curve(MTM~lp.3, data = dd.train, bootstraps = 50)
plot_decision_curve(list(dca.DLR, dca.CR, dca.3),
                    curve.names = c("DL radiomics-based nomogram", "Clinical-radiological model", "DL radiomics-based nomogram + Substantial Necrosis"),
                    col = c("#FF0000", "#0099FF", "#FFCC00"), 
                    confidence.intervals = F, ylim=c(0,1))

# mtext(expression(paste("(A)",sep="")),outer=F,at=0,cex=1.0)

# # compare A-NBC
# areadiff<-function(data, ii, frm1, frm2, start, stop, step){
#   data_sub<- data[ii, ]
#   dca1<- decision_curve(frm1, data = data_sub, bootstraps = 50)
#   dca2<- decision_curve(frm2, data = data_sub, bootstraps = 50)
#   
#   nb1=dca1$derived.data$sNB[start:stop]
#   nb2=dca2$derived.data$sNB[start:stop]
#   area1<-0;area2<-0
#   for(i in 1:(stop-start)){
#     area1<-area1+(nb1[i]+nb1[i+1])*step/2
#     area2<-area2+(nb2[i]+nb2[i+1])*step/2
#   }
#   cat('.')
#   return(area2-area1)
# }
# #=========
# set.seed(128);
# R<-100
# xstart<-21
# xstop<-90
# boot.area<-boot(data=dd.train, statistic=areadiff, R=R, frm1=MTM~lp.DLR, frm2=MTM~lp.CR,start=xstart, stop=xstop, step=0.01)
# glopvalue <-mean(abs(boot.area$t-boot.area$t0)>abs(boot.area$t0))
# cat("\n", "global p-value over threshold probabilities", xstart,'-',xstop,'=',glopvalue,'\n')
# # =======


dca.DLR <- decision_curve(MTM~lp.DLR, data = dd.internal_test, bootstraps = 50)
dca.CR <- decision_curve(MTM~lp.CR, data = dd.internal_test, bootstraps = 50)
dca.3 <- decision_curve(MTM~lp.3, data = dd.internal_test, bootstraps = 50)
plot_decision_curve(list(dca.DLR, dca.CR, dca.3),
                    curve.names = c("DL radiomics-based nomogram", "Clinical-radiological model", "DL radiomics-based nomogram + Substantial Necrosis"),
                    col = c("#FF0000", "#0099FF", "#FFCC00"), 
                    confidence.intervals = F, ylim=c(0,1))
# # =======
# R<-100
# xstart<-21
# xstop<-90
# boot.area<-boot(data=dd.internal_test, statistic=areadiff, R=R, frm1=MTM~lp.DLR, frm2=MTM~lp.CR, start=xstart, stop=xstop, step=0.01)
# glopvalue <-mean(abs(boot.area$t-boot.area$t0)>abs(boot.area$t0))
# cat("\n", "global p-value over threshold probabilities", xstart,'-',xstop,'=',glopvalue,'\n')
# # =======
# mtext(expression(paste("(B)",sep="")),outer=F,at=0,cex=1.0)


dca.DLR <- decision_curve(MTM~lp.DLR, data = dd.external_test, bootstraps = 50)
dca.CR <- decision_curve(MTM~lp.CR, data = dd.external_test, bootstraps = 50)
dca.3 <- decision_curve(MTM~lp.3, data = dd.external_test, bootstraps = 50)
plot_decision_curve(list(dca.DLR, dca.CR, dca.3),standardize = TRUE,
                    curve.names = c("DL radiomics-based nomogram", "Clinical-radiological model", "DL radiomics-based nomogram + Substantial Necrosis"),
                    col = c("#FF0000", "#0099FF", "#FFCC00"),  
                    confidence.intervals = F, ylim=c(0,1))
# #==========
# R<-100
# xstart<-21
# xstop<-90
# boot.area<-boot(data=dd.external_test, statistic=areadiff, R=R, frm1=MTM~lp.DLR, frm2=MTM~lp.CR, start=xstart, stop=xstop, step=0.01)
# glopvalue <-mean(abs(boot.area$t-boot.area$t0)>abs(boot.area$t0))
# cat("\n", "global p-value over threshold probabilities", xstart,'-',xstop,'=',glopvalue,'\n')
# # =======
# mtext(expression(paste("(C)",sep="")),outer=F,at=0,cex=1.0)
dev.off()

#----------
# ROC subgroup
#----------	

library(data.table)
library(ResourceSelection)
library(pROC)

dd = data.frame(fread("datainfo.csv", header = TRUE))[,-2]
listVars = colnames(dd)[-c(1:2)]
para = listVars[-26]

formul.DLR = as.formula(paste("MTM~",paste(c("AFP","AP.VMI","AP.ElectronDensity","PVP.IodineDensity"), collapse = '+'),sep=""))
fit.multi.DLR = glm(formul.DLR, binomial(logit), data = dd.train)
dd$lp.DLR = predict(fit.multi.DLR, newdata = dd,type = "response")
formul.CR = as.formula(paste("MTM~",paste(c("AFP","Obvineco"), collapse = '+'),sep=""))
fit.multi.CR = glm(formul.CR, binomial(logit), data = dd.train)
dd$lp.CR = predict(fit.multi.CR, newdata = dd,type = "response")
dd.train = dd[which(dd$dataset == 1),]
dd.internal_test = dd[which(dd$dataset == 2),]
dd.external_test = dd[which(dd$dataset == 0),]

#	Size
dd.train.size1 = dd.train[which(dd.train$size==1),]
dd.train.size0 = dd.train[which(dd.train$size==0),]
dd.internal_test.size1 = dd.internal_test[which(dd.internal_test$size==1),]
dd.internal_test.size0 = dd.internal_test[which(dd.internal_test$size==0),]
dd.external_test.size1 = dd.external_test[which(dd.external_test$size==1),]
dd.external_test.size0 = dd.external_test[which(dd.external_test$size==0),]

auc.1a = roc(dd.train.size1$MTM, dd.train.size1$lp.DLR) 
auc.2a = roc(dd.train.size0$MTM, dd.train.size0$lp.DLR) 
auc.3a = roc(dd.internal_test.size1$MTM, dd.internal_test.size1$lp.DLR) 
auc.4a = roc(dd.internal_test.size0$MTM, dd.internal_test.size0$lp.DLR) 
auc.5a = roc(dd.external_test.size1$MTM, dd.external_test.size1$lp.DLR) 
auc.6a = roc(dd.external_test.size0$MTM, dd.external_test.size0$lp.DLR) 

roc.PP.1 = roc.test(auc.1a,auc.2a)
roc.PP.2 = roc.test(auc.3a,auc.4a)
roc.PP.3 = roc.test(auc.5a,auc.6a)

#	subnecro
dd.train.Obvineco1 = dd.train[which(dd.train$Obvineco==1),]
dd.train.Obvineco0 = dd.train[which(dd.train$Obvineco==0),]
dd.internal_test.Obvineco1 = dd.internal_test[which(dd.internal_test$Obvineco==1),]
dd.internal_test.Obvineco0 = dd.internal_test[which(dd.internal_test$Obvineco==0),]
dd.external_test.Obvineco1 = dd.external_test[which(dd.external_test$Obvineco==1),]
dd.external_test.Obvineco0 = dd.external_test[which(dd.external_test$Obvineco==0),]

auc.1b = roc(dd.train.Obvineco1$MTM, dd.train.Obvineco1$lp.DLRE) 
auc.2b = roc(dd.train.Obvineco0$MTM, dd.train.Obvineco0$lp.DLR)
auc.3b = roc(dd.internal_test.Obvineco1$MTM, dd.internal_test.Obvineco1$lp.DLR)
auc.4b = roc(dd.internal_test.Obvineco0$MTM, dd.internal_test.Obvineco0$lp.DLR) 
auc.5b = roc(dd.external_test.Obvineco1$MTM, dd.external_test.Obvineco1$lp.DLR)
auc.6b = roc(dd.external_test.Obvineco0$MTM, dd.external_test.Obvineco0$lp.DLR) 

#----------
# KM
#----------

library(survival)
library(data.table)		

dd = data.frame(fread("dd_survive.csv", header = TRUE))[,-1]
dd.train = dd[which(dd$dataset == 1),]
dd.internal_test = dd[which(dd$dataset == 2),]
dd.external_test = dd[which(dd$dataset == 0),]

library(pROC)
dd.train$RFS = dd.train$RFS/30
dd.internal_test$RFS = dd.internal_test$RFS/30
dd.external_test$RFS = dd.external_test$RFS/30
library(survminer)

dd$Group = factor(cut(dd$lp.DLR,
                            breaks = c(-10,summary(dd$lp)[3],10),
                            labels = c("Low risk","High risk")))
fit <- survfit(Surv(RFS, Recurence) ~ Group,
               data = dd)



dd.train$Group = factor(cut(dd.train$lp.DLR, 
                            breaks = c(-10,summary(dd.train$lp)[3],10),
                            labels = c("Low risk","High risk")))

# write.csv(dd.train, "train_rfs.csv", row.names = TRUE)
fit <- survfit(Surv(RFS, Recurence) ~ Group,  
               data = dd.train) 


tiff(paste("KM train.tiff",sep=""),width = 3000, height = 2200,res=300)
gg = ggsurvplot(fit, 
                data = dd.train, 
                #conf.int = TRUE, 
                pval = TRUE, 
                #surv.median.line = "hv",  
                risk.table = TRUE, 
                
                xlab = "Follow up Time (Month)", 
                legend = c(0.8,0.3), 
                legend.title = "", #
                #legend.labs = c("Not Lost", "Lost"), 
                break.x.by = 4) 
print(gg)
dev.off()

dd.internal_test$Group = factor(cut(dd.internal_test$lp.DLR, 
                           breaks = c(-10,summary(dd.internal_test$lp)[3],10),
                           labels = c("Low risk","High risk")))

fit <- survfit(Surv(RFS,Recurence) ~ Group,  
               data = dd.internal_test)
# write.csv(dd.internal_test, "internal_test_rfs.csv", row.names = TRUE)

tiff(paste("KM internal test.tiff",sep=""), width = 3000, height = 2200, res=300)
gg = ggsurvplot(fit, 
                data = dd.internal_test,  
                #conf.int = TRUE, 
                pval = TRUE, 
                #surv.median.line = "hv",  
                risk.table = TRUE, 
                xlab = "Follow up Time (Month)", 
                legend = c(0.8,0.3), 
                legend.title = "", 
                #legend.labs = c("Not Lost", "Lost"),
                break.x.by = 4)  
print(gg)
dev.off()


dd.external_test$Group = factor(cut(dd.external_test$lp.DLR, 
                                 breaks = c(-10,summary(dd.external_test$lp)[3],10),
                                 labels = c("Low risk","High risk")))

fit <- survfit(Surv(RFS,Recurence) ~ Group,  
               data = dd.external_test) 

# write.csv(dd.external_test, "external_test_rfs.csv", row.names = TRUE)

tiff(paste("KM external test.tiff",sep=""),width = 3000, height = 2200, res=300)
gg = ggsurvplot(fit,
                data = dd.external_test,
                # conf.int = TRUE,
                pval = TRUE,
                # surv.median.line = "hv",
                risk.table = TRUE,
                xlab = "Follow up Time (Month)",
                legend = c(0.8,0.3),
                legend.title = "",
                # legend.labs = c("Not Lost", "Lost"),
                break.x.by = 4)

print(gg)
dev.off()


