library(tidyverse)
library(tidymodels)
d <- read_csv("lab-1/data/ngsschat-processed-data.csv")
d
dim(d)
ls(d)

# split the data
train_test_split <- initial_split(d, prop = .70)
data_train <- training(train_test_split)
data_test <- testing(train_test_split)

# Engineer features
my_rec <- recipe(code ~ ., data = data_train)
my_rec

## Specify recipe, model, and workflow
# specify model
my_mod <-
    logistic_reg() %>% 
    set_engine("glm") %>%
    set_mode("classification")

# specify workflow
my_wf <-
    workflow() %>%
    add_model(my_mod) %>% 
    add_recipe(my_rec)

# Fit model
fitted_model <- fit(my_wf, data = data_train)
final_fit <- last_fit(fitted_model, train_test_split)

fitted_model

# Evaluate accuracy
final_fit %>%
    collect_metrics

# Random Forest Tree - Lab 4

install.packages("vip")
library(tidyverse)
library(tidymodels)
library(vip) # a new package we're adding for variable importance measures

getwd()
setwd("C:/Users/yzh7381/Documents/GitHub/machine-learning/lab-4")


d <- read_csv("data/ngsschat-processed-data-add-three-features.csv")

train_test_split <- initial_split(d, prop = .80)
data_train <- training(train_test_split)
kfcv <- vfold_cv(data_train, v = 5) # again, we will use resampling

my_rec <- recipe(code ~ ., data = data_train) %>% 
    step_normalize(all_numeric_predictors()) %>%
    step_nzv(all_predictors())


# specify model

install.packages("ranger")

my_mod <-
    rand_forest(mtry = tune(), # this specifies that we'll take steps later to tune the model
                min_n = tune(),
                trees = tune()) %>%
    set_engine("ranger", importance = "impurity") %>% #importance is to check which variables do a better job to split the data. The benefit of RF is to attain the variable importance.
    set_mode("classification")

# specify workflow
my_wf <-
    workflow() %>%
    add_model(my_mod) %>% 
    add_recipe(my_rec)
my_wf

# specify tuning grid
finalize(mtry(), data_train)#in the output, show the range, 1 to 8, choose variables from 1 to 8
finalize(min_n(), data_train)
finalize(trees(), data_train)

tree_grid <- grid_max_entropy(mtry(range(1, 18)), # there's different combination of these three parameters, 1, 2, 1
                              min_n(range(2, 40)),
                              trees(range(1, 600)), # how to determine how many trees are enough? can go back and forth, or see the 1st five rows. If they're close enough. Would stop.
                              size = 10) # combination of the tuning parameters
tree_grid

# fit model with tune_grid
fitted_model <- my_wf %>% 
    tune_grid(
        resamples = kfcv,
        grid = tree_grid,
        metrics = metric_set(roc_auc, accuracy, kap, sensitivity, specificity, precision))# specify the accuracy metrics here
        

# examine best set of tuning parameters; repeat?
show_best(fitted_model, n = 10, metric = "accuracy")#can look at the 1st three columns to check the 3 tuning parameters, try to find the best tuning parameters, the mean column is the average accuracy, may add the sd of mean?

# select best set of tuning parameters
best_tree <- fitted_model %>% select_best(metric = "accuracy")
best_tree

# finalize workflow with best set of tuning parameters
final_wf <- my_wf %>% 
    finalize_workflow(best_tree)

final_fit <- final_wf %>% 
    last_fit(train_test_split, metrics = metric_set(roc_auc, accuracy, kap, sensitivity, specificity, precision))

collect_metrics(final_fit) 




