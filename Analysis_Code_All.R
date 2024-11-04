#2020302131253肖皓天
#Econometrics Game

#It should be higgggggghlighted that there are some hyperparameter optimization operations in my code
#The outcome of them might be random due to the algorithm, so the best param maybe different when run at next time

#IMPORT DATA AND PACKAGE 
#There are some special packages that I found in the github
library(haven)
library(dplyr)
library(caret)
library(mlbench)
library(glmnet)
library(mice)
library(ggcorrplot)
library(ggplot2)
library(GGally)
library(scales)
library(memisc)
library(corrplot)
library(PerformanceAnalytics)
library(knitr)
library(stargazer)
library(tidyverse)
library(broom)
library(ivreg)
library(lmtest)
library(sandwich)
library(MASS)
library(ranger)
library(rBayesianOptimization)
library(mlr3verse)
library(xgboost)
library(SHAPforxgboost)
library(data.table)
library(here)
library(plm)
library(VGAM)
library(ordinal)
library(brant)
library(rJava)
library(devtools)
library(usethis)
library(rJava)
library(instruments)#devtools::install_github("alexpavlakis/instruments")


data1 <- read_sav("D:/GSS2012.sav")
data2 <- read_sav("D:/GSS2014.sav")
data3 <- read_sav("D:/GSS2016.sav")

#################################################################################################################################
#################################################################################################################################


#Question1 
#Combine the data and delete some features
GSSdata<-bind_rows(data1,data2,data3)
missing <- colSums(is.na(GSSdata))
missing_pct <- missing / nrow(GSSdata)
to_remove <- which(missing_pct > 0.3)
GSSdata <- GSSdata[,-to_remove , drop=FALSE]
#Use random forest to fill in the NAs, it should be highlighted that I use rf because some features are "int" and rf will not fill in decimal fraction
imputed_data1 <- mice(GSSdata, method="rf")
GSSdata <- complete(imputed_data1)
#RF may not fill in all the NAs so omit NAs
GSSdata <- na.omit(GSSdata)

#Use LASSO to select the features related to HAPPY, it should be mentioned that there are various methods to do feature engineering
#For instance, Ridge regression, Elastic Net, RFE, Blend, Null-Importance, for me I love lasso most so I choose it
index0 <-  sort(sample(nrow(GSSdata), nrow(GSSdata)*.7))
train0 <- GSSdata[index0,]
test0 <-  GSSdata[-index0,]
#Divide the train set and test set
x0 <- subset(train0,select=-c(HAPPY))
y0 <- subset(train0, select=c(HAPPY))
#Y should be number or cannot put into the lasso regression
y0 <- as.numeric(y0$HAPPY)
lasso <- glmnet(x0, y0, family = "gaussian", alpha = 1)
print(lasso)
#Plot the lasso result and coefficient
plot(lasso, xvar = "lambda", label = TRUE)
lasso.coef <- coef(lasso, s = 0.045)
print(lasso.coef)
#X should be number or cannot put into the lasso regression
x0 <- as.matrix(x0)
#Plot the cv resulty
cvfit=cv.glmnet(x0,y0)
plot(cvfit)
#Print best lambda
cvfit$lambda.1se
#Find the best model from two lambda
l.coef1<-coef(cvfit$glmnet.fit,s=0.02579153,exact = F)
l.coef1
#I choose the variable whose abs(coefficient) is larger than e-03, only INCOME is related to HAPPY in general 
#Bar chart of INCOME VS HAPPY
GSSdata <- subset(GSSdata, select=c(HAPPY,INCOME,YEAR))
GSSdata$INCOME=factor(GSSdata$INCOME)
data_grouped <- GSSdata %>%
  group_by(INCOME) %>%
  summarise(mean_HAPPY = mean(HAPPY))
ggplot(data = data_grouped, aes(x = INCOME, y = mean_HAPPY, fill = INCOME)) +
  geom_bar(stat = "identity") +
  xlab("INCOME") +
  ylab("Mean HAPPY") +
  ggtitle("Relationship between INCOME and HAPPY") +
  scale_x_discrete()

#For the number of samples whose income=1 is extremely small, that can be seen as noise
#Conclude that with income increasing, the HAPPY level increasing at first but with income increasing to above 8 the happy level is decreasing
#Other factors should be taken into account
#The proof of in general TECH and MEDIA do not significantly correlate with HAPPY will be proved using GSS2014 and GSS2016

#################################################################################################################################
#################################################################################################################################

#Question 2
#USETECH is what I attach most significance to, so first delete those USETECH = NA
gssdata2 <- data2
gssdata2 <- gssdata2[!is.na(gssdata2$USETECH),]

#Then delete the features whose NA exceeds 30%
missing <- colSums(is.na(gssdata2))
missing_pct <- missing / nrow(gssdata2)
to_remove <- which(missing_pct > 0.3)
gssdata2 <- gssdata2[,-to_remove , drop=FALSE]

#Use random forest to fill in the NAs
imputed_data2 <- mice(gssdata2, method="rf")
gssdata2 <- complete(imputed_data2)
#RF may not fill in all the NAs so omit NAs
gssdata2 <- na.omit(gssdata2)

#Use USETECH to represent technology, explore the relationship between HAPPY INCOME and TECHNOLOGY
GSScor <- subset(gssdata2, select=c('INCOME','USETECH','HAPPY'))
chart.Correlation(GSScor,histogram = TRUE,pch=19)
#From the chart it is obvious that TECH is weakly correlated with HAPPY in general pattern

#First I do an order logit regression
#Use LASSO to select the features related to HAPPY
index <-  sort(sample(nrow(gssdata2), nrow(gssdata2)*.7))
train <- gssdata2[index,]
test <-  gssdata2[-index,]
#Divide the train set and test set
x2 <- subset(train,select=-c(HAPPY,USETECH))
y2 <- subset(train, select=c(HAPPY))
#Y should be number or cannot put into the lasso regression
y2 <- as.numeric(y2$HAPPY)
lasso2 <- glmnet(x2, y2, family = "gaussian", alpha = 1)
#X should be number or cannot put into the lasso regression
x2 <- as.matrix(x2)
#Plot the cv result
cvfit2=cv.glmnet(x2,y2)
#Print best lambda
cvfit2$lambda.min
#Find the best model from lambda
l.coef2<-coef(cvfit2$glmnet.fit,s=0.03226389,exact = F)
l.coef2

#Use LASSO to select the features related to USETECH
index <-  sort(sample(nrow(gssdata2), nrow(gssdata2)*.7))
train <- gssdata2[index,]
test <-  gssdata2[-index,]
#Divide the train set and test set
x3 <- subset(train,select=-c(HAPPY,USETECH))
y3 <- subset(train, select=c(USETECH))
#Y should be number or cannot put into the lasso regression
y3 <- as.numeric(y3$USETECH)
lasso3 <- glmnet(x3, y3, family = "gaussian", alpha = 1)
#X should be matrix or cannot put into the lasso regression
x3 <- as.matrix(x2)
#CV
cvfit3=cv.glmnet(x3,y3)
#Print best lambda
cvfit3$lambda.min 
#Find the best model from lambda
l.coef3<-coef(cvfit3$glmnet.fit,s=1.160755,exact = F)
l.coef3

#Combine model 1 and model 2, select the features by theory and economic intuition
#Select WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE as control variable
#Order logit regression
gssdata2A <- gssdata2
gssdata2A$HAPPY <- factor(gssdata2A$HAPPY)
q2model1 <- polr(HAPPY ~  USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE, data = gssdata2A, method="logistic")
summary(q2model1)
brant(q2model1)
#Omnibus p=0.44>0.05, cannot reject H0,Order Logit Regression true
#Some other test
drop1(q2model1,test="Chi") 
q2model1a <- polr(HAPPY ~  1, data = gssdata2A, method="logistic")
q2model1b <- polr(HAPPY ~  USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE, data = gssdata2A, method="logistic")
anova(q2model1a,q2model1b)

#I separate HAPPY level into low, middle and high and do logit regression one by one
gssdata2B <- gssdata2
gssdata2B$LHAPPY[gssdata2B$HAPPY != 1] <- 0
gssdata2B$LHAPPY[gssdata2B$HAPPY == 1] <- 1
gssdata2B$MHAPPY[gssdata2B$HAPPY != 2] <- 0
gssdata2B$MHAPPY[gssdata2B$HAPPY == 2] <- 1
gssdata2B$HHAPPY[gssdata2B$HAPPY != 3] <- 0
gssdata2B$HHAPPY[gssdata2B$HAPPY == 3] <- 1
#REGION FE MODEL
gssdata2B <- as.data.frame(gssdata2B)
gssdata2B<-pdata.frame(gssdata2B,index = "REGION")
#Low happy
lmodel1<-plm(LHAPPY ~  USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE,data=gssdata2B,model="within",family = binomial(link = "logit"))
summary(lmodel1)
#Middle happy
lmodel2<-plm(MHAPPY ~  USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE,data=gssdata2B,model="within",family = binomial(link = "logit"))
summary(lmodel2)
#High happy
lmodel3<-plm(HHAPPY ~  USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE,data=gssdata2B,model="within",family = binomial(link = "logit"))
summary(lmodel3)
#From the outcome, USETECH is only significant with high happy level

#So third I do 2SLS
#When it comes to the choice of iv, I do think that the outcome of feature selection by lasso will be plausible
#To be more specific, the features selected by USETECH lasso but not selected by HAPPY lasso is what I attach significance to
#It can be seen as instrument variable because it correlates with USETECH but not correlates with HAPPY

#Select the instrument iv by theory and economic intuition
#Select LEARNNEW as instrument variable
#Intuition: At this technology era, when people is learning new things in work, it is likely to be related to technology, or, he will use some technology like computer to learn it;

#Logit IV
#I use a package from github https://github.com/alexpavlakis/instruments
#Divide HAPPY into low=1,middle=2,high=3 
#The package only support one instrumental variable, so I choose LEARNNEW, which is intuitively more valid
#For low Happy
gssdata2a <- gssdata2
gssdata2a$HAPPY <- as.numeric(as.character(gssdata2a$HAPPY))
gssdata2a$HAPPY[gssdata2a$HAPPY != 1] <- 0
glm_iv <- iv.glm(model_formula = HAPPY ~ USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE, 
                 instrument_formula = USETECH ~ LEARNNEW, 
                 data=gssdata2a, family = binomial, link = 'logit')
summary(glm_iv)
#From the summary USETECH is a little significant in low HAPPY level
diagnose(glm_iv)
#No return, the instrumental variable is valid

#For middle Happy
gssdata2b <- gssdata2
gssdata2b$HAPPY <- as.numeric(as.character(gssdata2b$HAPPY))
gssdata2b$HAPPY[gssdata2b$HAPPY != 2] <- 0
gssdata2b$HAPPY[gssdata2b$HAPPY == 2] <- 1
glm_iv2 <- iv.glm(model_formula = HAPPY ~ USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE, 
                 instrument_formula = USETECH ~ LEARNNEW, 
                 data=gssdata2b, family = binomial, link = 'logit')
summary(glm_iv2)
#From the summary USETECH is not significant in middle HAPPY level
diagnose(glm_iv2)
#No return, the instrumental variable is valid

#For high HAPPY
gssdata2c <- gssdata2
gssdata2c$HAPPY <- as.numeric(as.character(gssdata2c$HAPPY))
gssdata2c$HAPPY[gssdata2c$HAPPY != 3] <- 0
gssdata2c$HAPPY[gssdata2c$HAPPY == 3] <- 1
glm_iv3 <- iv.glm(model_formula = HAPPY ~ USETECH+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE, 
                  instrument_formula = USETECH ~ LEARNNEW, 
                  data=gssdata2c, family = binomial, link = 'logit')
summary(glm_iv3)
#From the summary USETECH is not significant in high HAPPY level
diagnose(glm_iv3)
#No return, the instrumental variable is valid

#Third, I consider the effect of INCOME on happy, divide the different income group and do logit regression
gssdata2B$INCOME <- as.numeric(as.character(gssdata2B$INCOME))
gssdata2B$INC_group1 <- ifelse(gssdata2B$INCOME %in% 1:3, 1, 0)
gssdata2B$INC_group2 <- ifelse(gssdata2B$INCOME %in% 4:6, 1, 0)
gssdata2B$INC_group3 <- ifelse(gssdata2B$INCOME %in% 7:9, 1, 0)
gssdata2B$INC_group4 <- ifelse(gssdata2B$INCOME %in% 10:12, 1, 0)
#Create Interaction terms
gssdata2B$INC1 <- gssdata2B$INC_group1 * gssdata2B$USETECH
gssdata2B$INC2 <- gssdata2B$INC_group2 * gssdata2B$USETECH
gssdata2B$INC3 <- gssdata2B$INC_group3 * gssdata2B$USETECH
gssdata2B$INC4 <- gssdata2B$INC_group4 * gssdata2B$USETECH
#Logit regression
lmodel4<-plm(HHAPPY ~  INC1+INC2+INC3+INC4+WORKDIFF+USEDUP+COWRKHLP+INTETHN+HEFINFO+JOBSECOK+RACLIVE,data=gssdata2B,model="within",family = binomial(link = "logit"))
summary(lmodel4)

#Finally, I turn to machine learning to solve both question.
#1.USETECH significant or not? 2.INCOME has effect or not?
#I establish a BO-XGBOOST-SHAP explainable machine learning model
#BO is bayes optimization which performs parameters optimization
#XGBOOST is which I use as the regressor
#Considering the maching learning model is like a black box which is hard to interpret to human
#I adopt the SHAP method to make explainable maching learning model

#INCOME with USETECH
gssdata2c$INCOME <- as.numeric(as.character(gssdata2c$INCOME))
gssdata2c$INC_group1 <- ifelse(gssdata2c$INCOME %in% 1:3, 1, 0)
gssdata2c$INC_group2 <- ifelse(gssdata2c$INCOME %in% 4:6, 1, 0)
gssdata2c$INC_group3 <- ifelse(gssdata2c$INCOME %in% 7:9, 1, 0)
gssdata2c$INC_group4 <- ifelse(gssdata2c$INCOME %in% 10:12, 1, 0)

gssdata2c$INC1 <- gssdata2c$INC_group1 * gssdata2c$USETECH
gssdata2c$INC2 <- gssdata2c$INC_group2 * gssdata2c$USETECH
gssdata2c$INC3 <- gssdata2c$INC_group3 * gssdata2c$USETECH
gssdata2c$INC4 <- gssdata2c$INC_group4 * gssdata2c$USETECH

#Divide the train and test set
label <-gssdata2c$HAPPY
gssxgboost <- as.matrix(gssdata2c[,c("USETECH","WORKDIFF","USEDUP","COWRKHLP","INTETHN","HEFINFO","JOBSECOK","RACLIVE","INC1","INC2","INC3","INC4")])
dtrain <- xgb.DMatrix(data = gssxgboost ,label = label)
cv_folds <- KFold(label, nfolds = 5, stratified = TRUE, seed = 909)

#Objective Function & Model
xgb_cv_bayes <- function(max_depth, min_child_weight, subsample) {
  cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
                             max_depth = as.integer(max_depth),
                             min_child_weight = as.integer(min_child_weight),
                             subsample = subsample, 
                             objective = "reg:squarederror",
                             eval_metric = "rmse"),
               data = dtrain, nround = 100,
               folds = cv_folds, prediction = TRUE, showsd = TRUE,
               early_stopping_rounds = 5, maximize = FALSE, verbose = 0)
  list(Score = cv$evaluation_log$test_rmse_mean[cv$best_iteration],
       Pred = cv$pred)
}

#Bayesian Optimization
OPT_Res <- BayesianOptimization(xgb_cv_bayes,
                                bounds = list(max_depth = c(2L, 6L),
                                              min_child_weight = c(1L, 10L),
                                              subsample = c(0.5, 0.8)),
                                init_grid_dt = NULL, init_points = 10, n_iter = 20,
                                acq = "ucb", kappa = 2.576, eps = 0.0,
                                verbose = TRUE)
#Best Parameters Found: Round = 26	max_depth = 6.0000	min_child_weight = 1.0000	subsample = 0.8000	Value = 0.3329594 

#SHAP to explain the machine learning model
mod <- xgboost::xgboost(data = dtrain, 
                        label = label, 
                        params = list(max_depth = 6.0000	,min_child_weight = 1.0000,	subsample = 0.8000), nrounds = 10,
                        verbose = FALSE, nthread = parallel::detectCores() - 2,
                        early_stopping_rounds = 5)

#Create the summary plot
#The summary plot shows global feature importance. The sina plots show the distribution of feature contributions to the model output
shap_values <- shap.values(xgb_model = mod, X_train = dtrain)
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = gssxgboost)
shap.plot.summary(shap_long)

#Creat SHAP force plot
#The SHAP force plot basically stacks these SHAP values for each observation
#And show how the final output was obtained as a sum of each predictor’s attributions.
plot_data <- shap.prep.stack.data(shap_contrib = shap_values$shap_score, top_n = 4, n_groups = 6)
shap.plot.force_plot_bygroup(plot_data)

#From the outcome of machine learning, we can learn that USETECH has a relatively big effect on HAPPINESS
#At the same time, INC4 truly has an effect on the happiness
#However, from the distribution of INCOME, the samples INCOME in 10-12 takes up the most percentage
#So it is hard to say whether high income matters

#################################################################################################################################
#################################################################################################################################

#Question3 
#Internet Time is what I attach great significance to, so I create a SOCMEDIA = INTWKENM
#Choose weekend is because in this fast-pace and busy era, only the social media used in weekend is important
#Intuitively, in weekdays many people use social media for work
gssdata3 <- data3
gssdata3$SOCMEDIA <- gssdata3$INTWKENM
gssdata3 <- gssdata3[, -c(which(colnames(gssdata3) == "INTWKENM"))]
gssdata3 <- gssdata3[!is.na(gssdata3$SOCMEDIA),]
#Delete the NA
missing <- colSums(is.na(gssdata3))
missing_pct <- missing / nrow(gssdata3)
to_remove <- which(missing_pct > 0.3)
gssdata3 <- gssdata3[,-to_remove , drop=FALSE]
#Fill in the NA
imputed_data3 <- mice(gssdata3, method="rf")
gssdata3 <- complete(imputed_data3)
sum(is.na(gssdata3))
gssdata3 <- na.omit(gssdata3)
#Draw a correlation picture of INCOME SOCMEDIA HAPPY
GSScor2 <- subset(gssdata3, select=c('INCOME','SOCMEDIA','HAPPY'))
chart.Correlation(GSScor2,histogram = TRUE,pch=19)
#From the picture, HAPPY is weakly correlated with INCOME and SOCMEDIA in general pattern

#Use lasso to select the features correlated to HAPPY as Q2
index <-  sort(sample(nrow(gssdata3), nrow(gssdata3)*.7))
train <- gssdata3[index,]
test <-  gssdata3[-index,]
#Divide the train set and test set
x4 <- subset(train,select=-c(HAPPY,SOCMEDIA,INTWKDYM,INTWKENH,INTWKDYH))
y4 <- subset(train, select=c(HAPPY))
#Y should be number or cannot put into the lasso regression
y4 <- as.numeric(y4$HAPPY)
lasso3 <- glmnet(x4, y4, family = "gaussian", alpha = 1)
#X should be number or cannot put into the lasso regression
x4 <- as.matrix(x4)
#Plot the cv result
cvfit4=cv.glmnet(x4,y4)
#Print best lambda
cvfit4$lambda.min
#Find the best model from lambda
l.coef4<-coef(cvfit4$glmnet.fit,s=0.05136836,exact = F)
l.coef4

#Use lasso to select the features correlated to SOCMEDIA as Q2
index <-  sort(sample(nrow(gssdata3), nrow(gssdata3)*.7))
train <- gssdata3[index,]
test <-  gssdata3[-index,]
#Divide the train set and test set
x5 <- subset(train,select=-c(HAPPY,SOCMEDIA,INTWKDYM,INTWKENH,INTWKDYH))
y5 <- subset(train, select=c(SOCMEDIA))
#Y should be number or cannot put into the lasso regression
y5 <- as.numeric(y5$SOCMEDIA)
lasso5 <- glmnet(x5, y5, family = "gaussian", alpha = 1)
#X should be number or cannot put into the lasso regression
x5 <- as.matrix(x5)
#Plot the cv result
cvfit5=cv.glmnet(x5,y5)
#Print best lambda
cvfit5$lambda.min 
#Find the best model from lambda
l.coef5<-coef(cvfit5$glmnet.fit,s=0.190119,exact = F)
l.coef5
#Combine model 1 and model 2, select the features by PDS theory
#Select HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN as regression variable

#Order Logit Regression
GSSdata3 <-gssdata3
GSSdata3$HAPPY <- factor(GSSdata3$HAPPY)
q3model2 <- polr(HAPPY ~  SOCMEDIA+HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN, data = GSSdata3, method="logistic")
summary(q3model2)
brant(q3model2)
#Omnibus p=0.47>0.1,cannot reject H0,Order Logit Regression can be used
#Some other tests
drop1(q3model2,test="Chi") 
q3model2a <- polr(HAPPY ~  1, data = GSSdata3, method="logistic")
q3model2b <- polr(HAPPY ~  SOCMEDIA+ARTEXBT+CONDOM+COOP+COURTS+INTRECNT+FAIR, data = GSSdata3, method="logistic")
anova(q3model2a,q3model2b)
#From the outcome, SOCMEDIA is not significant

#Logit Regression with fixed region
gssdata3B <- gssdata3
gssdata3B$LHAPPY[gssdata3B$HAPPY != 1] <- 0
gssdata3B$LHAPPY[gssdata3B$HAPPY == 1] <- 1
gssdata3B$MHAPPY[gssdata3B$HAPPY != 2] <- 0
gssdata3B$MHAPPY[gssdata3B$HAPPY == 2] <- 1
gssdata3B$HHAPPY[gssdata3B$HAPPY != 3] <- 0
gssdata3B$HHAPPY[gssdata3B$HAPPY == 3] <- 1
#REGION FE MODEL
gssdata3B <- as.data.frame(gssdata3B)
gssdata3B<-pdata.frame(gssdata3B,index = "REGION")
#Low happy
lmodel4<-plm(LHAPPY ~  SOCMEDIA+HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN,data=gssdata3B,model="within",family = binomial(link = "logit"))
summary(lmodel4)
#Middle happy
lmodel5<-plm(MHAPPY ~  SOCMEDIA+HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN,data=gssdata3B,model="within",family = binomial(link = "logit"))
summary(lmodel5)
#High happy
lmodel6<-plm(HHAPPY ~  SOCMEDIA+HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN,data=gssdata3B,model="within",family = binomial(link = "logit"))
summary(lmodel6)
#From the outcome, SOCMEDIA is not significant

#IV 2SLS
#Select TWITTER as the instrumental variable, which is intuitively
#Do 2SLS
#For low Happy
gssdata3a <- gssdata3
gssdata3a$HAPPY <- as.numeric(as.character(gssdata3a$HAPPY))
gssdata3a$IV <- gssdata3a$SNAPCHAT+gssdata3a$FACEBOOK+gssdata3a$TWITTER+gssdata3a$FLICKR+gssdata3a$TUMBLR+gssdata3a$INSTAGRM+gssdata3a$LINKEDIN+gssdata3a$CLSSMTES+gssdata3a$VINE+gssdata3a$WHATSAPP+gssdata3a$GOOGLESN           
gssdata3a$HAPPY[gssdata3a$HAPPY != 1] <- 0
glm_iv4 <- iv.glm(model_formula = HAPPY ~ SOCMEDIA+HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN, 
                 instrument_formula = SOCMEDIA ~ IV, 
                 data=gssdata3a, family = binomial, link = 'logit')
summary(glm_iv4)
#From the summary USETECH seems not significant in low HAPPY level
diagnose(glm_iv4)
#No return, the instrumental variable is valid

#For middle Happy
gssdata3b <- gssdata3
gssdata3b$HAPPY <- as.numeric(as.character(gssdata3b$HAPPY))
gssdata3b$IV <- gssdata3b$SNAPCHAT+gssdata3b$FACEBOOK+gssdata3b$TWITTER+gssdata3b$FLICKR+gssdata3b$TUMBLR+gssdata3b$INSTAGRM+gssdata3b$LINKEDIN+gssdata3b$CLSSMTES+gssdata3b$VINE+gssdata3b$WHATSAPP+gssdata3b$GOOGLESN           
gssdata3b$HAPPY[gssdata3b$HAPPY != 2] <- 0
gssdata3b$HAPPY[gssdata3b$HAPPY == 2] <- 1
glm_iv5 <- iv.glm(model_formula = HAPPY ~ SOCMEDIA+HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN, 
                  instrument_formula = SOCMEDIA ~ IV, 
                  data=gssdata3b, family = binomial, link = 'logit')
summary(glm_iv5)
#From the summary USETECH seems not significant in low HAPPY level
diagnose(glm_iv5)
#No return, the instrumental variable is valid

#For high HAPPY
gssdata3c <- gssdata3
gssdata3c$HAPPY <- as.numeric(as.character(gssdata3c$HAPPY))
gssdata3c$IV <- gssdata3c$SNAPCHAT+gssdata3c$FACEBOOK+gssdata3c$TWITTER+gssdata3c$FLICKR+gssdata3c$TUMBLR+gssdata3c$INSTAGRM+gssdata3c$LINKEDIN+gssdata3c$CLSSMTES+gssdata3c$VINE+gssdata3c$WHATSAPP+gssdata3c$GOOGLESN           
gssdata3c$HAPPY[gssdata3c$HAPPY != 3] <- 0
gssdata3c$HAPPY[gssdata3c$HAPPY == 3] <- 1
glm_iv6 <- iv.glm(model_formula = HAPPY ~ SOCMEDIA+HARASS5+TUMBLR+MARCOHAB+PARTNRS5+SATFIN, 
                  instrument_formula = SOCMEDIA ~ IV, 
                  data=gssdata3c, family = binomial, link = 'logit')
summary(glm_iv6)
#From the summary USETECH seems significant in low HAPPY level
diagnose(glm_iv6)
#No return, the instrumental variable is valid


#Finally turn to machine learning
#Divide the train and test set
label2 <-gssdata3c$HAPPY
gssxgboost2 <- as.matrix(gssdata3c[,c("SOCMEDIA","HARASS5","TUMBLR","MARCOHAB","PARTNRS5","SATFIN")])
dtrain2 <- xgb.DMatrix(data = gssxgboost2 ,label = label2)
cv_folds <- KFold(label2, nfolds = 5, stratified = TRUE, seed = 909)

#Objective Function & Model
xgb_cv_bayes2 <- function(max_depth, min_child_weight, subsample) {
  cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
                             max_depth = as.integer(max_depth),
                             min_child_weight = as.integer(min_child_weight),
                             subsample = subsample, 
                             objective = "reg:squarederror",
                             eval_metric = "rmse"),
               data = dtrain2, nround = 100,
               folds = cv_folds, prediction = TRUE, showsd = TRUE,
               early_stopping_rounds = 5, maximize = FALSE, verbose = 0)
  list(Score = cv$evaluation_log$test_rmse_mean[cv$best_iteration],
       Pred = cv$pred)
}

#Bayesian Optimization
OPT_Res2 <- BayesianOptimization(xgb_cv_bayes2,
                                bounds = list(max_depth = c(2L, 6L),
                                              min_child_weight = c(1L, 10L),
                                              subsample = c(0.5, 0.8)),
                                init_grid_dt = NULL, init_points = 10, n_iter = 20,
                                acq = "ucb", kappa = 2.576, eps = 0.0,
                                verbose = TRUE)
#Best Parameters Found: Round = 21	max_depth = 6.0000	min_child_weight = 2.0000	subsample = 0.5099209	Value = 0.337982  
#SHAP to explain the machine learning model
mod2 <- xgboost::xgboost(data = dtrain2, 
                        label = label, 
                        params = list(max_depth = 4.0000	,min_child_weight = 3.0000,	subsample = 0.7199209), nrounds = 10,
                        verbose = FALSE, nthread = parallel::detectCores() - 2,
                        early_stopping_rounds = 5)

#Create the summary plot
#The summary plot shows global feature importance. The sina plots show the distribution of feature contributions to the model output
shap_values2 <- shap.values(xgb_model = mod2, X_train = dtrain2)
shap_long2 <- shap.prep(shap_contrib = shap_values2$shap_score, X_train = gssxgboost2)
shap.plot.summary(shap_long2)

#Create SHAP force plot
#The SHAP force plot basically stacks these SHAP values for each observation
#And show how the final output was obtained as a sum of each predictor’s attributions.
plot_data2 <- shap.prep.stack.data(shap_contrib = shap_values2$shap_score, top_n = 4, n_groups = 6)
shap.plot.force_plot_bygroup(plot_data2)
