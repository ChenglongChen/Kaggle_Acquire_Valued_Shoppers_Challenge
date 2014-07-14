
#####################################################################
# Description:
# Train GLM for repeater classification, grid search for the best
# lambda and class weight for glmnet, and make submission in the
# required csv format.
# 
# The best lambda and class weight found by grid search are then
# used as params to train the bagging version.
# 
# This code yields score around 0.609.
# 
# Author: Chenglong Chen < yr@Kaggle >
# Email: c.chenglong@gmail.com>
#####################################################################


rm(list=ls(all=TRUE))
gc(reset=TRUE)


## put all the packages here
require(glmnet)
require(data.table)
require(bit64)
require(caTools)
require(foreach)


#####################
## Helper function ##
#####################

#### This function imputes each feature with some constant values
Impute <- function(data, value){
  
  num <- apply(data, 2, function(x) sum(is.na(x)))
  
  data <- data.matrix(data)
  data[which(is.na(data))] <- rep(value, num)
  data <- as.data.frame(data)
  
  return(data)
}


#### This function trains bagging glm using data sampled with/without replacement
bagging_glm <-function(X_train, Y_train, X_test, glm.param, bagging.iterations=10,
                       sample_replace=TRUE, sample_ratio=1.0, feat_ratio=1.0, seed=2014){
  # ensure reproducible results
  set.seed(seed)
  # number of training data
  numTrain <- nrow(X_train)
  numFeat <- ncol(X_train)
  predictions <- foreach(m=seq(1,bagging.iterations),.combine=cbind) %do% {
    
    # sample data with replacement
    trainInd <- sample(numTrain, size=floor(numTrain*sample_ratio), replace=sample_replace)
    # sample features as well (without replacement as in Random Forest)
    if(feat_ratio==1.0){
      featInd <- seq(1, numFeat)
    }else{
      featInd <- sample(numFeat, size=floor(numFeat*feat_ratio), replace=FALSE)
    }    
    
    # train glm
    weights <- ifelse(Y_train[trainInd]==1, glm.param$weight, 1)
    model <- glmnet(x = X_train[trainInd,featInd],
                    y = Y_train[trainInd],
                    family = "binomial",
                    weights = weights,
                    alpha = glm.param$alpha,
                    lambda = glm.param$lambda)
    
    # make prediction
    predict(model, X_test, type = "response")
  }
  # average prediction
  rowMeans(predictions)
}


####################
## Initialization ##
####################

## set the working directory to the folder containing
# - ./Data/train_Jun27_shopper.csv
# - ./Data/test_Jun27_shopper.csv
# - ./Data/testHistory.csv
# - ./Data/trainHistory.csv
path_to_Acquire <- 'E:/Acquire'
setwd(path_to_Acquire)

## path to data
csvTrainfile <- './Data/train_Jun27_shopper.csv'
csvTestfile <- './Data/test_Jun27_shopper.csv'
testHistoryfile <- './Data/testHistory.csv'
trainHistoryfile <- './Data/trainHistory.csv'

## path to save csv submission
filePath_csv <- './Submission/GLM'
dir.create(filePath_csv, showWarnings=FALSE, recursive=TRUE)


##################
## Read in data ##
##################

#### read in history data
# training
dfTrainHistory <- read.csv(trainHistoryfile)
numTrain <- nrow(dfTrainHistory)
offer_date <- as.Date(dfTrainHistory$offerdate)
# test
dfTestHistory <- read.csv(testHistoryfile)
numTest <- nrow(dfTestHistory)


#### read in training data
dfTrain <- as.data.frame(fread(csvTrainfile))
#dfTrain <- read.csv(csvTrainfile,header=T)
dfTrain$offer_date <- as.Date(dfTrain$offer_date)
# missing data imputation
na.num <- apply(dfTrain, 2, function(x) sum(is.na(x)))
const <- rep(0,dim(dfTrain)[2])
dfTrain[,na.num>0] <- Impute(dfTrain[,na.num>0], const[na.num>0])


#### read in test data
dfTest <- as.data.frame(fread(csvTestfile))
# dfTest <- read.csv(csvTestfile,header=T)
dfTest$offer_date <- as.Date(dfTest$offer_date)
# get test ID
testID <- as.numeric(dfTest[,1])
# missing data imputation
na.num <- apply(dfTest, 2, function(x) sum(is.na(x)))
const <- rep(0,dim(dfTest)[2])
dfTest[,na.num>0] <- Impute(dfTest[,na.num>0], const[na.num>0])


#### convert to factor
factor.vars <- c('offer', 'market', 'company', 'category', 'brand',
                 'dept', 'offer_mday', 'offer_mweek', 'offer_weekday')

trnInd <- seq(1,dim(dfTrain)[1])
for(v in factor.vars){
  f <- as.factor(c(dfTrain[,v], dfTest[,v]))  
  dfTrain[,v] <- f[trnInd]
  dfTest[,v] <- f[-trnInd] 
}


###################################
## Perform training & validation ##
###################################

# get the names of all predictors
unused <- c( 'id', 'repeater', 'repeattrips', 'offer_date',
             'offer', 'market', 'company', 'category', 'brand', 'dept', 'chain', 'mday', 'mweek', 'days', 'weekday')
unused <- c( 'id', 'repeater', 'repeattrips', 'offer_date', 'offer_mday', 'offer_days', 'offer_weekday')
#unused <- c( 'id', 'repeater', 'repeattrips', 'offer_date')
all.vars <- names(dfTrain)
# DEL <- grepl('^overall', all.vars)
# unused <- c(unused, all.vars[DEL])
predictors <- all.vars[-which(all.vars %in% unused)]


## bulid design matrix and response matrix
formula <- repeater ~ .
D_train <- model.frame(formula, data = dfTrain[,c(predictors, 'repeater')]) 
X_train <- model.matrix(formula, data = D_train) 
Y_train <- model.response(D_train)
rm(D_train)
gc(reset=TRUE)
## bulid design matrix
D_test <- model.frame(formula, data = dfTest[,c(predictors, 'repeater')]) 
X_test <- model.matrix(formula, data = D_test) 
rm(D_test)
gc(reset=TRUE)


## split into training and validation set according to offer date
trainRatio <- 0.6
thresh <- sort(dfTrain$offer_date)[floor(trainRatio*dim(dfTrain)[1])]
thresh <- '2013-04-01'
# indices for training and validation set
table(dfTrain$days)
# we elimate of days of 11 which seem outliers
outlier <- -1
trainInd <- which(dfTrain$offer_date<thresh & dfTrain$offer_days>outlier)
validInd <- which(dfTrain$offer_date>=thresh)


## grid search for the best lambda and class weight
# params grid 
glm.lambdas <- 2^seq(17,17,1) # large lambda works best
glm.alphas <- 0 # 0 works best
glm.weights <- 2^seq(-5,1,0.1)

best_auc <- rep(0, length(glm.weights))
glm.best.lambda <- rep(0, length(glm.weights))
glm.best.alpha <- rep(0, length(glm.weights))
count <- 0
for(glm.weight in glm.weights){
  count <- count + 1
  # sample weights
  weights <- ifelse(Y_train[trainInd]==1, glm.weight, 1)
  auc <- matrix(0, length(glm.lambdas), length(glm.alphas))
  for(i in seq(1,length(glm.lambdas))){
    glm.lambda <- glm.lambdas[i]
    for(j in seq(1,length(glm.alphas))){
      glm.alpha <- glm.alphas[j]
      
      # train glm
      model <- glmnet(x = X_train[trainInd,],
                      y = Y_train[trainInd],                
                      family = "binomial",
                      weights = weights,
                      alpha = glm.alpha,
                      lambda = glm.lambda)
      
      # make prediction
      y_valid_pred <- predict(model, X_train[validInd,], type = "response")
      auc[i,j] <- colAUC(y_valid_pred, Y_train[validInd])
    }
  }
  best_auc[count] <- max(auc)
  ind <- which(auc == best_auc[count], arr.ind=TRUE)
  r <- ind[1]
  c <- ind[1]
  glm.best.lambda[count] <- glm.lambdas[r]
  glm.best.alpha[count] <- glm.alphas[c]
  cat('Weight = ', glm.weight,      
      ' | lambda = ', glm.best.lambda[count],
      ' | alpha = ',glm.best.alpha[count],
      ' | AUC = ', best_auc[count],
      '\n', sep='')
}

best_AUC <- max(best_auc)
ind <- which.max(best_auc)
glm.best.weight <- glm.weights[ind]
glm.best.lambda <- glm.best.lambda[ind]
glm.best.alpha <- glm.best.alpha[ind]


cat('Best AUC = ', best_AUC,
    ' with weight = ', glm.best.weight,
    ', lambda = ', glm.best.lambda,
    ' and alpha = ', glm.best.alpha, sep='')


######################################################
## Train final model & Make prediction & submission ##
######################################################

## retrain the final model
# we elimate of days of 10 which seem outliers
trainInd <- which(dfTrain$offer_days>outlier)
# sample weights
weights <- ifelse(Y_train[trainInd]==1, glm.best.weight, 1)
model <- glmnet(x = X_train[trainInd,],
                y = Y_train[trainInd],
                family = "binomial",
                weights = weights,
                alpha = glm.best.alpha,
                lambda = glm.best.lambda)

# make prediction
p <- predict(model, X_test, type = "response")
## The predictions for IDs not existing in test.csv are 0.
pred <- rep(0, numTest)
pred[dfTestHistory$id %in% testID] <- p


## write a submission file. Leaderboard Public: 0.60468
sub <- data.frame(id = dfTestHistory$id,
                  repeatProbability = pred)

fileName <- paste(filePath_csv, '/GLM_',
                  '[Feature_Jun27]_',
                  '[Alpha',glm.best.alpha,']_',
                  '[Lambda',glm.best.lambda,']_',
                  '[Weight',round(glm.best.weight,5),']_',
                  '[AUC', round(best_AUC,5), ']',
                  '.csv', sep='')
write.csv(sub, fileName, quote = FALSE, row.names=FALSE)



###########################
## Train bagging version ##
###########################
# params
glm.param <- list(lambda=glm.best.lambda,
                  alpha=glm.best.alpha,
                  weight=glm.best.weight)

bagging.iterations <- 10
sample_replace <- TRUE
sample_ratio <- 1.0
feat_ratio <- 1.0

# compute validation AUC
y_valid_pred <- bagging_glm(X_train=X_train[trainInd,],
                            Y_train=Y_train[trainInd],
                            X_test=X_train[validInd,],
                            glm.param=glm.param,
                            bagging.iterations=bagging.iterations,                            
                            sample_replace=sample_replace,
                            sample_ratio=sample_ratio,
                            feat_ratio=feat_ratio)

auc <- colAUC(y_valid_pred, Y_train[validInd])
cat('Bagging AUC: ', auc, '\n', sep='')


# generate final bagging prediction
trainInd <- which(dfTrain$offer_days>outlier)
p <- bagging_glm(X_train=X_train[trainInd,],
                 Y_train=Y_train[trainInd],
                 X_test=X_test,
                 glm.param=glm.param,
                 bagging.iterations=bagging.iterations,                            
                 sample_replace=sample_replace,
                 sample_ratio=sample_ratio,
                 feat_ratio=feat_ratio)

## The predictions for IDs not existing in test.csv are 0.
pred <- rep(0, numTest)
pred[dfTestHistory$id %in% testID] <- p


## write a submission file. Leaderboard Public: 0.60468
sub <- data.frame(id = dfTestHistory$id,
                  repeatProbability = pred)

fileName <- paste(filePath_csv, '/GLM_',
                  '[Feature_Jun27]_',
                  '[Bagging',bagging.iterations,']_',
                  '[Alpha',glm.alpha,']_',
                  '[Lambda',glm.lambda,']_',
                  '[Weight',round(glm.weight,5),']_',
                  '[AUC', round(auc,5), ']',
                  '.csv', sep='')
write.csv(sub, fileName, quote = FALSE, row.names=FALSE)
gc(reset=TRUE)