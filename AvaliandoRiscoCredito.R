# ---------------------------------------------------------------------------------------
# [R] - Avaliando Risco de Crédito - 07/07/2019
#
# Lucas Cesar Fernandes Ferreira
# WebSite : www.lucascesarfernandes.com.br
# Linkedin : https://www.linkedin.com/in/lucas-cesar-fernandes/
# E-mail : lucascesar270392@gmail.com
# Contato : (31) 9 8219-8765
# ---------------------------------------------------------------------------------------
# :: DETALHES ::
# Nesse projeto vamos tentar avaliar o risco de crédito com dados da German Credit Data.
# No dataset há 1000 observações e 21 variáveis nossa variável target é a credit.rating.
# ---------------------------------------------------------------------------------------

# Instalando os pacotes caso já não estejam instalando
if (! "caret" %in% installed.packages()) install.packages("caret")
if (! "gmodels" %in% installed.packages()) install.packages("gmodels")
if (! "randomForest" %in% installed.packages()) install.packages("randomForest")
if (! "Metrics" %in% installed.packages()) install.packages("Metrics")
if (! "DMwR" %in% installed.packages()) install.packages("DMwR")
if (! "ggplot2" %in% installed.packages()) install.packages("ggplot2")

# Carregando Pacotes
library(caret)
library(gmodels)
library(randomForest)
library(Metrics)
library(DMwR)
library(ggplot2)

# Definindo o diretório de trabalho
getwd()
setwd("C:/Users/Dell/Documents/Projetos-em-R/AnaliseRiscoCredito")

# Carregando os dados
dados <- read.csv('credit_dataset.csv', header = TRUE, sep = ',')
View(dados)
head(dados)

# Análise descritiva
dim(dados)
str(dados)

summary(dados)

# Tabela de frequência da target
table(dados$credit.rating)

# Verificando valores missing's
sapply(dados, function(x) sum(is.na(x)))

# Criando as funções para a tranformação das variáveis
to.factor <- function(df, variables){
  for(variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

scale.features <- function(df, variables){
  for(variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

# Transformando as variáveis para o tipo fator
categorical <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                 'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                 'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                 'other.credits', 'apartment.type', 'bank.credits', 'occupation',
                 'dependents', 'telephone', 'foreign.worker')

dados <- to.factor(dados, categorical)

# Normalizando as variáveis numéricas
summary(dados[c('credit.duration.months', 'age', 'credit.amount')])

numerics <- c("credit.duration.months", "age", "credit.amount")

dados <- scale.features(dados, numerics)
summary(dados[c('credit.duration.months', 'age', 'credit.amount')])
View(dados[c('credit.duration.months', 'age', 'credit.amount')])

# Criando de dados de treino e teste
set.seed(2019)
split <- createDataPartition(y = dados$credit.rating, p = 0.7, list = FALSE)
dados_train <- dados[split,]
dados_test <- dados[-split,]

# Seleção de variáveis
feature.selection <- function(n_iters=20, feature, class){
  set.seed(10)
  
  variable.size <- 1:10
  
  control <- rfeControl(functions = rfFuncs, method = "cv",
                        verbose = FALSE, returnResamp = "all",
                        number = n_iters)
  results.rfe <- rfe(x = feature, y = class,
                     sizes = variable.size,
                     rfeControl = control)
  
  return(results.rfe)
}

# Executando a função de Seleção de variáveis
results_rfe <- feature.selection(feature = dados_train[,-1],
                                 class = dados_train[,1])
results_rfe
varImp(results_rfe)

# Criando um modelo com a regressão logística glm
set.seed(2019)
model <- glm(credit.rating ~
               account.balance +
               credit.duration.months +
               previous.credit.payment.status +
               credit.amount +
               savings, data = dados_train, family = 'binomial')
summary(model)

model_predict <- predict(model, dados_test[,-1], type = 'response')
model_predict <- round(model_predict)

CrossTable(model_predict, dados_test$credit.rating, prop.chisq = FALSE)
confusionMatrix(table(data = model_predict, reference = dados_test$credit.rating), positive = '1')

# Tirando o valor da curva ROC
auc(dados_test$credit.rating, model_predict)

# Vamos criar outro modelo com a variável target balanceada
set.seed(2019)
dados_smote <- dados
dados_smote <- SMOTE(credit.rating ~ ., dados_smote,
                     perc.over = 200, k = 5, perc.under = 200)

round(prop.table(table(dados_smote$credit.rating))*100, digits = 2)
dim(dados_smote)
View(dados_smote)

# Criando dados de treino e teste
set.seed(2019)
split <- createDataPartition(y = dados_smote$credit.rating, p = 0.7, list = FALSE)

dados_train_smote <- dados_smote[split,]
dados_test_smote <- dados_smote[-split,]

dim(dados_train_smote)
round(prop.table(table(dados_train_smote$credit.rating))*100, digits = 2)

dim(dados_test_smote)
round(prop.table(table(dados_test_smote$credit.rating))*100, digits = 2)

# Executando a função de Seleção de variáveis
results_rfe_smote <- feature.selection(feature = dados_train_smote[,-1],
                                       class = dados_train_smote[,1])
results_rfe_smote
varImp(results_rfe_smote)

# Criando um modelo com a regressão logística glm
set.seed(2019)
model_smote <- glm(credit.rating ~., data = dados_train_smote, family = 'binomial')
summary(model_smote)

model_predict_smote <- predict(model_smote, dados_test_smote[,-1], type = 'response')
model_predict_smote <- round(model_predict_smote)

CrossTable(model_predict_smote, dados_test_smote$credit.rating, prop.chisq = FALSE)
confusionMatrix(table(data = model_predict_smote, reference = dados_test_smote$credit.rating), positive = '1')

# Criando um modelo com randomForest
model_smote_rf <- randomForest(credit.rating ~
                                 account.balance +
                                 credit.duration.months +
                                 savings +
                                 previous.credit.payment.status +
                                 age +
                                 credit.amount +
                                 other.credits +
                                 employment.duration +
                                 credit.purpose +
                                 current.assets +
                                 installment.rate, dados_train_smote)

model_predict_smote_rf <- predict(model_smote_rf, dados_test_smote[,-1])
confusionMatrix(model_predict_smote_rf, dados_test_smote$credit.rating, positive = '1')


# Criando um modelo com Gradient Boosting
set.seed(2019)
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

model_smote_gbm <- train(credit.rating ~ ., dados_train_smote,
                         method = 'gbm',
                         tuneGrid = gbmGrid,
                         trControl = trainControl(method = 'cv',
                                                  number =  10))

model_predict_smote_gbm <- predict(model_smote_gbm, dados_test_smote[,-1])
confusionMatrix(model_predict_smote_gbm, dados_test_smote$credit.rating, positive = '1')

 # Plotando resultados
trellis.par.set(caretTheme())
ggplot(model_smote_gbm)

# Criando mais dois modelos com os algoritmos SVM e RDA

# SVM
set.seed(2019)
model_smote_svm <- train(credit.rating ~ ., dados_train_smote, 
                         method = "svmRadial", 
                         trControl = trainControl(method = 'cv',
                                                  number =  10), 
                         preProc = c("center", "scale"),
                         tuneLength = 8)

model_predict_smote_svm <- predict(model_smote_svm, dados_test_smote[,-1])
confusionMatrix(model_predict_smote_svm, dados_test_smote$credit.rating, positive = '1')                

# RDA
set.seed(2019)
model_smote_rda <- train(credit.rating ~ ., dados_train_smote, 
                         method = "rda", 
                         trControl = trainControl(method = 'cv',
                                                  number =  10), 
                         tuneLength = 4)

model_predict_smote_rda <- predict(model_smote_rda, dados_test_smote[,-1])
confusionMatrix(model_predict_smote_rda, dados_test_smote$credit.rating, positive = '1')

# Neural Network
set.seed(2019)
model_smote_neural <- train(credit.rating ~ ., dados_train_smote, 
                         method = "nnet", 
                         trControl = trainControl(method = 'cv',
                                                  number =  10), 
                         tuneLength = 4)

model_predict_smote_neural <- predict(model_smote_neural, dados_test_smote[,-1])
confusionMatrix(model_predict_smote_neural, dados_test_smote$credit.rating, positive = '1')

# Análisando os modelos gbm, svm e rda
models.resamps <- resamples(list(GBM = model_smote_gbm,
                                 SVM = model_smote_svm,
                                 RDA = model_smote_rda,
                                 NET = model_smote_neural))
models.resamps
summary(models.resamps)

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(models.resamps)


