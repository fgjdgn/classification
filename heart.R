heart=read.csv( "https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
colSums(is.na(heart))
table(heart$fatal_mi)
heart$fatal_mi <- factor(heart$fatal_mi)
set.seed(123)
heart_train <- heart[sample.int(nrow(heart),200),]
table(heart_train$fatal_mi)
set.seed(123)
heart_test <- heart[-sample.int(nrow(heart),200),]
table(heart_test$fatal_mi)
cor_heart=cor(heart[,-length(heart)])
library("lattice")
levelplot(as.matrix(cor_heart))
summary(heart_train)
library("skimr")
skim(heart)
DataExplorer::plot_boxplot(heart_train, by = "fatal_mi")
DataExplorer::plot_bar(heart_train)
DataExplorer::plot_histogram(heart_train)

library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3viz)
library("data.table")
library("mlr3verse")

task <- TaskClassif$new(id = "Heart",
                               backend = heart_train, # <- NB: no na.omit() this time
                               target = "fatal_mi",
                               positive = "1")
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(task)
set.seed(123)
learner_lda = lrn("classif.lda", predict_type = "prob")
learner_cart = lrn("classif.rpart", predict_type = "prob")
learner_rf = lrn("classif.ranger", predict_type = "prob")
learner_xgboost = lrn("classif.xgboost", predict_type = "prob")
set.seed(123)       
res <- benchmark(data.table(
  task       = list(task),
  learner    = list(learner_lda,
                    learner_cart,
                    learner_rf,
                    learner_xgboost),
  resampling = list(cv5)
), store_models = TRUE)


res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

autoplot(res, measure = msr("classif.ce"), facet_cols = "learner_id")
autoplot(res, measure = msr("classif.acc"), facet_cols = "learner_id")
autoplot(res, measure = msr("classif.auc"), facet_cols = "learner_id")
autoplot(res, measure = msr("classif.fpr"), facet_cols = "learner_id")
autoplot(res, measure = msr("classif.fnr"), facet_cols = "learner_id")
autoplot(res, type = "roc", facet_cols = "learner_id")
autoplot(res, type = "prc", facet_cols = "learner_id")


library(ggplot2)
df <- as.data.frame(res$aggregate(list(msr("classif.ce"),
                                       msr("classif.acc"),
                                       msr("classif.auc"),
                                       msr("classif.fpr"),
                                       msr("classif.fnr"))))
df.id = df$learner_id
df = df[,7:11]
df = cbind(df,df.id)
df_long <- reshape2::melt(df)

ggplot(df_long, aes(x = variable, y = value, group = df.id, color = df.id)) +
  geom_line() +
  labs(title = "Comparison of Classification Metrics",
       x = "Metrics", y = "Score")

set.seed(123)
ps <- ps(
  mtry = p_int(1, ncol(heart)-1),
  splitrule = p_fct(c("gini", "extratrees")),
  min.node.size = p_int(1, 10),
  sample.fraction = p_dbl(0.5, 1),
  num.trees = p_int(1, 1000)
)

library(mlr3tuning)
evals20 = trm("evals", n_evals = 20)
set.seed(123)
instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner_rf,
  resampling = cv5,
  measure = msr("classif.ce"),
  search_space = ps,
  terminator = evals20
)
tuner = tnr("grid_search", resolution = 5)
set.seed(123)
result = tuner$optimize(instance)
best_param = instance$result_learner_param_vals
best_param
learner_rf = lrn("classif.ranger", predict_type = "prob")
set.seed(123)
learner_rf$param_set$values <- instance$result_learner_param_vals
set.seed(123)
learner_rf$train(task)
pred = predict(learner_rf, heart_test, predict_type = "prob")


calibration_data <- data.frame(prob = pred[, 1], label = heart_test$fatal_mi)
calibration_task <- TaskClassif$new("calibration", calibration_data, target = "label")
calibration_learner <- lrn("classif.log_reg", predict_type = "prob")
calibration_learner$train(calibration_task)
calibration_pred = predict(calibration_learner, calibration_data, predict_type = "prob")
fit_calibration_data = cbind(pred[, 1], calibration_pred[,2])
ggplot(as.data.frame(fit_calibration_data), aes(x = V1, y = V2)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Calibration Plot", x = "Original Probability", y = "Calibrated Probability")








