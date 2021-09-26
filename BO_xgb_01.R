library(DiceKriging)
library(mlrMBO)
library(xgboost)


set.seed(17)

library( "data.table")
library("ggplot2")

carpeta_datasetsOri <-  "../datasetsOri/"
septiembre <- "paquete_premium_202009.csv"

ds <- fread(paste0(carpeta_datasetsOri, septiembre,collapse = ""), header=TRUE, showProgress = FALSE)
ds$clase_binaria <- factor(ifelse(ds$clase_ternaria == "BAJA+2", 1, 0))
ds$clase_ternaria <- NULL

# Solo usaremos 5
semillas <- as.vector(unlist(fread("../cache/02_DT_semillas.txt")))[1:5]

################################################################################
# FUN
################################################################################

modelo_xgb <- function (train, test, md = 30, eta = 0.03) { 
  
  clases_train <- as.numeric(train$clase_binaria) - 1
  train$clase_binaria <- NULL
  train <- xgb.DMatrix(data = data.matrix(train),  label = clases_train, missing=NA )
  clases_train[clases_train == 0] = 'noevento'
  clases_train[clases_train == 1] = 'evento'
  
  clases_test <- as.numeric(test$clase_binaria) - 1
  test$clase_binaria <- NULL
  test <- xgb.DMatrix(data = data.matrix(test),  label = clases_test, missing=NA )
  clases_test[clases_test == 0] = 'noevento'
  clases_test[clases_test == 1] = 'evento'
  
  modelo <- xgb.train( 
    data = train,
    nround= 5, #20, # poner la mejor ronda
    objective="binary:logistic",
    verbose = 2,
    max_depth = md,
    eta = eta
  )
  
  test_prediccion <- predict(modelo, test , type = "prob")
  #test_prediction <- as.numeric(test_prediccion > 0.025)
  
  roc_pred <-  ROCR::prediction(test_prediccion, clases_test,
                                label.ordering=c("noevento", "evento"))
  auc_t <-  ROCR::performance( roc_pred,"auc")
  
  unlist(auc_t@y.values)
}


experimento_xgb <- function (ds, semillas, md = 30, eta = 0.3) {
  auc <- c()
  
  for (s in semillas) {
    set.seed(s)
    inTraining <- caret::createDataPartition(ds$clase_binaria, p = 0.70, list = FALSE)
    train  <-  ds[  inTraining, ]
    test   <-  ds[ -inTraining, ]
    
    #    r <- modelo_xgb(train[train_sample,], test,  cp = cp, ms = ms, mb = mb, md = md)
    
    r <- modelo_xgb(train, test, md = md, eta = eta)
    auc <- c(auc, r)
  }
  data.table(mean_auc = mean(auc), sd_auc = sd(auc))
}

################################################################################
# BO
################################################################################

obj.fun = makeSingleObjectiveFunction(
  name = "max depth & eta",
  fn = function(x) - experimento_xgb(ds, semillas, md = as.integer(x$maxdepth), eta = x$eta)$mean_auc,
  par.set = makeParamSet(makeNumericParam("maxdepth", lower = 4L, upper =  30L),
                         makeNumericParam("eta", lower = 0.01, upper = 1 )),
  has.simple.signature = FALSE
)


ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, iters = 10L)
ctrl = setMBOControlInfill(
  ctrl,
  crit = makeMBOInfillCritEI(),
  opt = "focussearch"
)

lrn = makeMBOLearner(ctrl, obj.fun)
design = generateDesign(4, getParamSet(obj.fun), fun = lhs::maximinLHS)

surr.km <- makeLearner("regr.km", predict.type = "se", covtype = "matern3_2")


run = exampleRun(
  obj.fun,
  design = design,
  learner = surr.km,
  control = ctrl,
  points.per.dim = 2,
  show.info = TRUE
)

saveRDS(run, "../cache/03_HO_md_OB.RDS")