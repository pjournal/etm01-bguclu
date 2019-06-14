require(data.table)
require(TunePareto)
require(glmnet)
require(caret)
require(randomForest)
require(rpart)
require(verification)
require(plyr)

setwd("C:\\Users\\burhan\\Desktop\\ETM 58D\\ETM 58D\\FinalProject")

testStart=as.Date('2018-08-01')
trainStart=as.Date('2010-08-01')
rem_miss_threshold=0.01 

source('data_preprocessing.r')
source('feature_extraction.r')
source('performance_metrics.r')
source('train_models.r')

matches_data_path='df9b1196-e3cf-4cc7-9159-f236fe738215_matches.rds'
odd_details_data_path='df9b1196-e3cf-4cc7-9159-f236fe738215_odd_details.rds'

# read data
matches_raw=readRDS(matches_data_path)
odd_details_raw=readRDS(odd_details_data_path)

# preprocess matches
matches=matches_data_preprocessing(matches_raw)

# preprocess odd data
odd_details=details_data_preprocessing(odd_details_raw,matches)

# extract open and close odd type features from multiple bookmakers
features=extract_features.openclose(matches,odd_details,pMissThreshold=rem_miss_threshold,trainStart,testStart)

# divide data based on the provided dates 
train_features=features[Match_Date>=trainStart & Match_Date<testStart] 
test_features=features[Match_Date>=testStart] 

##PRS Calculation function

rankProbScore <- function(predictions, observed){
  ncat <- ncol(predictions)
  npred <- nrow(predictions)
   
  rps <- numeric(npred)
   
  for (rr in 1:npred){
    obsvec <- rep(0, ncat)
    obsvec[observed[rr]] <- 1
    cumulative <- 0
    for (i in 1:ncat){
      cumulative <- cumulative + (sum(predictions[rr,1:i]) - sum(obsvec[1:i]))^2
    }
    rps[rr] <- (1/(ncat-1))*cumulative
  }
  return(rps)
}

observedResults=test_features[,ifelse(test_features$Match_Result=="Home",1,ifelse(test_features$Match_Result=="Tie",2,3))]
Id=c(1:350)

# GLMNET
predicted_probGLMNET=train_glmnet(train_features, test_features,not_included_feature_indices=c(1:5), 
alpha=1,nlambda=50, tune_lambda=TRUE,nofReplications=2,nFolds=10,trace=T)

predictionsGLMNETTable=as.data.table(cbind(Id,predictionsGLMNET))
predictionsGLMNETTable=predictionsGLMNETTable[,Max:=max(V2,V3,V4),by=Id]
predictionsGLMNETTable=predictionsGLMNETTable[,Prd:=ifelse(Max==V2,"Home",ifelse(Max==V4,"Away","Tie")),by=Id]
table(test_features$Match_Result,predictionsGLMNETTable$Prd)

predictionsGLMNET=cbind(predicted_probGLMNET$predictions$Home,predicted_probGLMNET$predictions$Tie,predicted_probGLMNET$predictions$Away)
PRSGLMNET=rankProbScore(predictionsGLMNET,observedResults)






not_included_feature_indices=c(1:5)

##XGB Model
nrounds=500
tune_grid <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = c(0.0025, 0.005),
  max_depth = c(4, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 5,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv",
  number = 5,  
  verboseIter = TRUE, 
  allowParallel = FALSE
)

xgb_tune <- caret::train(
  x = train_features[,-not_included_feature_indices,with=F],
  y = train_features$Match_Result,
  trControl = tune_control,
  tuneGrid = tune_grid,
  objective = 'multi:softprob',
  method = "xgbTree",
  verbose = TRUE
)

xgb_tune$bestTune

final_grid <- expand.grid(
  nrounds = xgb_tune$bestTune$nrounds,
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune$bestTune$max_depth,
  gamma = xgb_tune$bestTune$gamma,
  colsample_bytree = xgb_tune$bestTune$colsample_bytree,
  min_child_weight = xgb_tune$bestTune$min_child_weight,
  subsample = xgb_tune$bestTune$subsample
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, 
  allowParallel = TRUE, 
  classProbs=TRUE
)

xgb_model <- caret::train(x = train_features[,-not_included_feature_indices,with=F], y = train_features$Match_Result, 
                          trControl = train_control,
                          tuneGrid = final_grid,
                          method = "xgbTree",
                          verbose = TRUE,
                          objective = 'multi:softprob'
)

predictedXgb=predict(xgb_model, newdata = test_features)
table(test_features$Match_Result,predictedXgb)
predicted_probXgb=predict(xgb_model, newdata = test_features,type='prob')
predictionsXgb=cbind(predicted_probXgb$Home,predicted_probXgb$Tie,predicted_probXgb$Away)
PRSXgb=rankProbScore(predictionsXgb,observedResults)


##GBM

GBM_model <- train(x = train_features [,-not_included_feature_indices,with=F], y = train_features$Match_Result, 
             method     = "gbm",
             trControl  = tune_control)

predictedGBM=predict(GBM_model, newdata = test_features)
table(test_features$Match_Result,predictedGBM)
predicted_probGBM=predict(GBM_model, newdata = test_features,type="prob")
predictionsGBM=cbind(predicted_probGBM$Home,predicted_probGBM$Tie,predicted_probGBM$Away)
PRSGBM=rankProbScore(predictionsGBM,observedResults)

##Rpart
Rpart_model <- train(x = train_features [,-not_included_feature_indices,with=F], y = train_features$Match_Result, 
             method     = "rpart",
             trControl  = tune_control)


predictedRpart=predict(Rpart_model, newdata = test_features)
table(test_features$Match_Result,predictedRpart)
predicted_probRpart=predict(Rpart_model, newdata = test_features,type="prob")
predictionsRpart=cbind(predicted_probRpart$Home,predicted_probRpart$Tie,predicted_probRpart$Away)
PRSRpart=rankProbScore(predictionsRpart,observedResults)



##Random Forest
train_featuresRF=train_features[complete.cases(train_features)]
RF_model <- caret:: train(x = train_featuresRF [,-not_included_feature_indices,with=F], y = train_featuresRF$Match_Result, 
                              method = "rf",
					trControl  = tune_control,
					 metric="Accuracy",
                              verbose = TRUE,
					)


predictedRF=predict(RF_model, newdata = test_features)
table(test_features$Match_Result,predictedRF)
predicted_probRF=predict(RF_model, newdata = test_features,type='prob')
predictionsRF=cbind(predicted_probRF$Home,predicted_probRF$Tie,predicted_probRF$Away)
PRSRF=rankProbScore(predictionsRF,observedResults)

##KNN

train_featuresKnn=train_features[complete.cases(train_features)]
scaled=train_featuresKnn[,scale(train_featuresKnn[,-(1:5)])]

Knn_model <- train(x = train_featuresKnn [,-not_included_feature_indices,with=F], y = train_featuresKnn$Match_Result, 
             method     = "knn",
		tuneLength = 3,
             trControl  = tune_control)

Knn_model_scaled <- train(x = scaled, y = train_featuresKnn$Match_Result, 
             method     = "knn",
		tuneLength = 3,
             trControl  = tune_control)


predictedKnn=predict(Knn_model, newdata = test_features)
table(test_features$Match_Result,predictedKnn)
predicted_probKnn=predict(Knn_model, newdata = test_features,type="prob")
predictionsKnn=cbind(predicted_probKnn$Home,predicted_probKnn$Tie,predicted_probKnn$Away)
PRSKnn=rankProbScore(predictionsKnn,observedResults)

#Scaled KNN
scaledTest=test_features[,scale(test_features[,-(1:5)])]
predictedKnnScaled=predict(Knn_model_scaled, newdata = scaledTest)
table(test_features$Match_Result,predictedKnnScaled)
predicted_probKnn_scaled=predict(Knn_model_scaled, newdata = scaledTest,type="prob")
predictionsKnnScaled=cbind(predicted_probKnn_scaled$Home,predicted_probKnn_scaled$Tie,predicted_probKnn_scaled$Away)
PRSKnnScaled=rankProbScore(predictionsKnnScaled,observedResults)


##Probability Ranking Score Evaluation

MatchResults=test_features$Match_Result
PRSTable=cbind(Id,MatchResults,PRSGLMNET,PRSXgb,PRSRF,PRSGBM,PRSRpart,PRSKnn,PRSKnnScaled)
PRSTable=as.data.table(PRSTable)
PRSTable=PRSTable[,MinValue:=min(PRSGLMNET,PRSXgb,PRSRF,PRSGBM,PRSRpart,PRSKnn,PRSKnnScaled),by=Id]
PRSTable=PRSTable[,BestMethod:=ifelse(MinValue==PRSGLMNET,"PRSGLMNET",
					 ifelse(MinValue==PRSXgb,"PRSXgb",
					 ifelse(MinValue==PRSRF,"PRSRF",
					 ifelse(MinValue==PRSGBM,"PRSGBM",
					 ifelse(MinValue==PRSRpart,"PRSRpart",
					 ifelse(MinValue==PRSKnn,"PRSKnn",
					 ifelse(MinValue==PRSKnnScaled,"PRSKnnScaled",0))))))),by=Id]

PRSMethodTable=table(PRSTable$BestMethod)
a=PRSTable[,list(BestMethod),by=list(MatchResults)]
b=a[MatchResults=="Home"]
c=a[MatchResults=="Tie"]
d=a[MatchResults=="Away"]
home=table(b$BestMethod)
tie=table(c$BestMethod)
away=table(d$BestMethod)




#Without Scaled Knn

PRSTable2=cbind(Id,MatchResults,PRSGLMNET,PRSXgb,PRSRF,PRSGBM,PRSRpart,PRSKnn)
PRSTable2=as.data.table(PRSTable2)
PRSTable2=PRSTable2[,MinValue:=min(PRSGLMNET,PRSXgb,PRSRF,PRSGBM,PRSRpart,PRSKnn),by=Id]
PRSTable2=PRSTable2[,BestMethod:=ifelse(MinValue==PRSGLMNET,"PRSGLMNET",
					 ifelse(MinValue==PRSXgb,"PRSXgb",
					 ifelse(MinValue==PRSRF,"PRSRF",
					 ifelse(MinValue==PRSGBM,"PRSGBM",
					 ifelse(MinValue==PRSRpart,"PRSRpart",
					 ifelse(MinValue==PRSKnn,"PRSKnn",0)))))),by=Id]

PRSMethodTable2=table(PRSTable2$BestMethod)

a2=PRSTable2[,list(BestMethod),by=list(MatchResults)]
b2=a2[MatchResults=="Home"]
c2=a2[MatchResults=="Tie"]
d2=a2[MatchResults=="Away"]
home2=table(b2$BestMethod)
tie2=table(c2$BestMethod)
away2=table(d2$BestMethod)


##Without Knn

PRSTable3=cbind(Id,MatchResults,PRSGLMNET,PRSXgb,PRSRF,PRSGBM,PRSRpart,PRSKnnScaled)
PRSTable3=as.data.table(PRSTable3)
PRSTable3=PRSTable3[,MinValue:=min(PRSGLMNET,PRSXgb,PRSRF,PRSGBM,PRSRpart,PRSKnnScaled),by=Id]
PRSTable3=PRSTable3[,BestMethod:=ifelse(MinValue==PRSGLMNET,"PRSGLMNET",
					 ifelse(MinValue==PRSXgb,"PRSXgb",
					 ifelse(MinValue==PRSRF,"PRSRF",
					 ifelse(MinValue==PRSGBM,"PRSGBM",
					 ifelse(MinValue==PRSRpart,"PRSRpart",
					 ifelse(MinValue==PRSKnnScaled,"PRSKnnScaled",0)))))),by=Id]

PRSMethodTable3=table(PRSTable3$BestMethod)
a3=PRSTable3[,list(BestMethod),by=list(MatchResults)]
b3=a3[MatchResults=="Home"]
c3=a3[MatchResults=="Tie"]
d3=a3[MatchResults=="Away"]
home3=table(b3$BestMethod)
tie3=table(c3$BestMethod)
away3=table(d3$BestMethod)


