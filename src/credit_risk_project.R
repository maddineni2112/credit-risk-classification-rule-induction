
# Data Mining Project 1 Mile stone 1
#Loading the libraries
library(tidyverse)
library(mlbench)
library(rpart)
library(rpart.plot)
library(caret)
library(RWeka)
# install.packages(RWeka)
# install.packages(RWekajars)
#install.packages(Rtsne)
library(FSelector)
library(Rtsne) #used for reducing to two dimensions
library(ggplot2) ## used for plotting the tsne plot.
library(dplyr)

# Loading the  data
credit <- read.csv(
  "C:/Users/SampathNagaMaddineni/Downloads/datamining/credit-g.csv",
  stringsAsFactors = TRUE
)
# checking if data loaded correctly
cat("Head of Credit:\n")
print(head(credit))

#to set the uniformiry of the test train split class ratios
set.seed(123)

# changing Target as factor
credit$class <- as.factor(credit$class)

# Removing the  duplicates
credit_clean <- unique(credit)

#  Splitting the data in train-test (70/30 stratified) 
idx_train <- createDataPartition(credit_clean$class, p = 0.70, list = FALSE)
train_raw <- credit_clean[idx_train, ]
test_raw  <- credit_clean[-idx_train, ]

cat("Train rows:", nrow(train_raw), "\n")
cat("Test rows :", nrow(test_raw), "\n")

#-----------------
# Task 1 – Part 1: Prepossessing Steps
#-----------------

# 1) converting logical columns into factors
#----------------

train_raw <- train_raw %>%
  mutate(across(where(is.logical), ~ factor(.x, levels = c(TRUE, FALSE)))) %>%
  mutate(across(where(is.character), as.factor))

test_raw <- test_raw %>%
  mutate(across(where(is.logical), ~ factor(.x, levels = c(TRUE, FALSE)))) %>%
  mutate(across(where(is.character), as.factor))

# Convert integer-coded categorical columns to factor
#Here we are converting columns that have less than 10 unique values into factors
int_cols <- names(train_raw)[sapply(train_raw, is.integer)]
int_like_cat <- int_cols[sapply(train_raw[int_cols], function(x) dplyr::n_distinct(x, na.rm = TRUE) <= 10)]
int_like_cat <- setdiff(int_like_cat, "class")

if (length(int_like_cat) > 0) {
  train_raw[int_like_cat] <- lapply(train_raw[int_like_cat], factor)
  test_raw[int_like_cat]  <- lapply(test_raw[int_like_cat], factor)
  cat("Converted integer-coded categorical columns to factor:\n")
  print(int_like_cat)
} else {
  cat("No integer-coded categorical integer columns detected.\n")
}

# Align factor levels in test to match train
for (col in names(train_raw)) {
  if (is.factor(train_raw[[col]])) {
    test_raw[[col]] <- factor(test_raw[[col]], levels = levels(train_raw[[col]]))
  }
}

cat("\nSummary train_raw:\n")
print(summary(train_raw))
cat("\nSummary test_raw:\n")
print(summary(test_raw))

# 2) checking for Missing values
#---------------

cat("\nMissing values per column (TRAIN):\n")
print(colSums(is.na(train_raw)))

cat("\nMissing values per column (TEST):\n")
print(colSums(is.na(test_raw)))

cat("\nRows with any NA (TRAIN):", sum(apply(is.na(train_raw), 1, any)), "\n")
cat("Rows with any NA (TEST) :", sum(apply(is.na(test_raw), 1, any)), "\n")

# Drop NA rows (dataset typically has none, but included for completeness)
train_raw <- train_raw[!apply(is.na(train_raw), 1, any), ]
test_raw  <- test_raw[!apply(is.na(test_raw), 1, any), ]

cat("\nAfter dropping NA rows:\n")
cat("Train rows:", nrow(train_raw), "\n")
cat("Test rows :", nrow(test_raw), "\n")

# 3) Feature selection using Chi-Square (top 15)
#------------
N <- 15

weights <- chi.squared(class ~ ., train_raw)

weights_tbl <- as.data.frame(weights) %>%
  rownames_to_column("feature") %>%
  arrange(desc(attr_importance))

cat("\nChi-square feature importance (top rows):\n")
print(head(weights_tbl, 20))

subset <- cutoff.k(weights, N)
f <- as.simple.formula(subset, "class")

cat("\nSelected-feature formula f:\n")
print(f)

selected_features <- all.vars(f)
selected_features <- setdiff(selected_features, "class")

train_fs <- train_raw %>% select(all_of(selected_features), class)
test_fs  <- test_raw  %>% select(all_of(selected_features), class)

# Align factor levels again after feature selection
for (col in names(train_fs)) {
  if (is.factor(train_fs[[col]])) {
    test_fs[[col]] <- factor(test_fs[[col]], levels = levels(train_fs[[col]]))
  }
}

cat("\nSummary train_fs:\n")
print(summary(train_fs))

#-----------------
# Task 1 – Part 2: t-SNE Visualization on full dataset
#-----------------
credit_tsne <- credit_clean %>%
  mutate(class = as.factor(class)) %>%
  filter(complete.cases(.))

X <- credit_tsne %>% select(-class)
y <- credit_tsne$class

dmy <- dummyVars(~ ., data = X, fullRank = TRUE)
X_oh <- as.data.frame(predict(dmy, newdata = X))
X_scaled <- scale(X_oh)

set.seed(123)
tsne_out <- Rtsne(
  X_scaled,
  dims = 2,
  perplexity = 30,
  theta = 0.5
)

tsne_df <- data.frame(
  TSNE1 = tsne_out$Y[, 1],
  TSNE2 = tsne_out$Y[, 2],
  class = y
)

ggplot(tsne_df, aes(x = TSNE1, y = TSNE2, color = class)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal() +
  labs(
    title = "t-SNE on full dataset (one-hot + scaled)",
    x = "t-SNE Dimension 1",
    y = "t-SNE Dimension 2",
    color = "Class"
  )
#----------------------------------------------------------------------------------


#----------------
# Task 2 – Part 1: Building DT, PART, RIPPER with out cross validation.
#----------------

train_fs$class <- as.factor(train_fs$class)
test_fs$class  <- factor(test_fs$class, levels = levels(train_fs$class))

#1. Decision Tree using the method rpart
#---------
set.seed(123)
dt_fit <- rpart(
  class ~ .,
  data = train_fs,
  method = "class",
  control = rpart.control(cp = 0.001, minsplit = 10, xval = 10)
)

print("\n---- Decision Tree (rpart) summary ----\n")
print(summary(dt_fit))

print("\n---- DT CP table ----\n")
printcp(dt_fit)

# Prune using 1-SE rule
cp_tbl <- dt_fit$cptable
best_row <- which.min(cp_tbl[, "xerror"])
xerr_min <- cp_tbl[best_row, "xerror"]
xerr_se  <- cp_tbl[best_row, "xstd"]
cp_1se   <- max(cp_tbl[cp_tbl[, "xerror"] <= xerr_min + xerr_se, "CP"])

dt_pruned <- prune(dt_fit, cp = cp_1se)

print("\n---- Pruned DT CP used ----\n")
print(cp_1se)

rpart.plot(dt_pruned, main = "Pruned Decision Tree (rpart)")

dt_pred <- predict(dt_pruned, newdata = test_fs, type = "class")
print("\n---- DT Confusion Matrix (TEST) ----\n")
print(confusionMatrix(dt_pred, test_fs$class))

# 2.PART
#----------

set.seed(123)
part_fit <- RWeka::PART(class ~ ., data = train_fs)
print("\n---- PART model ----\n")
print(part_fit)

part_pred <- predict(part_fit, newdata = test_fs)
print("\n---- PART Confusion Matrix (TEST) ----\n")
print(confusionMatrix(part_pred, test_fs$class))

# 3. RIPPER (JRip)
#------------------
set.seed(123)
ripper_fit <- RWeka::JRip(class ~ ., data = train_fs)
cat("\n---- RIPPER (JRip) model ----\n")
print(ripper_fit)

ripper_pred <- predict(ripper_fit, newdata = test_fs)
cat("\n---- RIPPER Confusion Matrix (TEST) ----\n")
print(confusionMatrix(ripper_pred, test_fs$class))


#----------------
# Task 2 – Part 2: Extract rule bases and saving to seperate files
#----------------

out_dir <- "task2_rule_bases"
if (!dir.exists(out_dir)) dir.create(out_dir)
#decision tree
dt_rules_txt <- capture.output({
  cat("Decision Tree Rules (from pruned rpart model)\n")
  cat("================================================\n\n")
  rpart.rules(dt_pruned, roundint = FALSE)
})
writeLines(dt_rules_txt, file.path(out_dir, "DT_rules.txt"))
#Part
part_rules_txt <- capture.output({
  cat("PART Rule Base\n")
  cat("==============\n\n")
  print(part_fit)
})
writeLines(part_rules_txt, file.path(out_dir, "PART_rules.txt"))
#Ripper
ripper_rules_txt <- capture.output({
  cat("RIPPER (JRip) Rule Base\n")
  cat("=======================\n\n")
  print(ripper_fit)
})
writeLines(ripper_rules_txt, file.path(out_dir, "RIPPER_rules.txt"))

print("\n Saved Task 2 rule bases in folder in  out_dir \n")

# -----------------
# Task 3: Part1 : Apply CV and reimplementing all three models
# ---------------

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = "final")

set.seed(3333)

dtree_cv <- train(form = f, data = train_fs, method = "rpart", parms = list(split = "information"), trControl = trctrl, tuneLength = 5)

part_cv <- train( form = f, data = train_fs, method = "PART", trControl = trctrl,tuneLength = 5)

jrip_cv <- train(form = f, data = train_fs, method = "JRip", trControl = trctrl, tuneLength = 5
)

#  compute 30 accuracies from $pred (per Resample)
get_fold_acc_from_pred <- function(fit_obj) {
  
  # caret stores out-of-fold predictions here when savePredictions="final"
  pred_df <- fit_obj$pred
  
  
  # Compute accuracy per resample fold:
  acc_by_fold <- pred_df %>%
    group_by(Resample) %>%
    summarise(Accuracy = mean(pred == obs), .groups = "drop") %>%
    arrange(Resample)
  
  return(acc_by_fold$Accuracy)
}

acc_dt   <- get_fold_acc_from_pred(dtree_cv)
acc_part <- get_fold_acc_from_pred(part_cv)
acc_jrip <- get_fold_acc_from_pred(jrip_cv)

print("\n Decision Tree  Accuracy Vector:\n")
print(acc_dt)
print("\n PART  Accuracy Vector:\n")
print(acc_part)
print("\n JRip  Accuracy Vector:\n")
print(acc_jrip)

print("\nFold accuracy vector lengths (must be 30):\n")
print(c(DT = length(acc_dt), PART = length(acc_part), JRip = length(acc_jrip)))

print("\nMean accuracies:\n")
print(c(DT = mean(acc_dt), PART = mean(acc_part), JRip = mean(acc_jrip)))


#---------------
#TASk2 Part2: ANOVA (F-test) and ANOVA TABLE
# -----------------
acc_df <- data.frame(
  Accuracy = c(acc_dt, acc_part, acc_jrip),
  Model = factor(rep(c("DecisionTree", "PART", "JRip"), each = 30))
)

anova_fit <- aov(Accuracy ~ Model, data = acc_df)

print("\n ------------- ANOVA TABLE ------------------- \n")
anova_out <- summary(anova_fit)
print(anova_out)

#----------------
# TASK3 - PART03: Tukey only if significant
#----------------
p_val <- anova_out[[1]][["Pr(>F)"]][1]

if (!is.na(p_val) && p_val < 0.05) {
  
  cat("\nANOVA: significant difference (p < 0.05). Running Tukey...\n")
  tuk <- TukeyHSD(anova_fit)
  print(tuk)
  
  mean_by_model <- acc_df %>%
    group_by(Model) %>%
    summarise(mean_accuracy = mean(Accuracy), .groups = "drop") %>%
    arrange(desc(mean_accuracy))
  
  cat("\nMean accuracy by model:\n")
  print(mean_by_model)
  
  cat("\nHighest mean accuracy model:\n")
  print(mean_by_model$Model[1])
  
} else {
  cat("\nANOVA: NOT significant (p >= 0.05). No difference among classifiers.\n")
}



#######################################################################################################

# MILESTONE 2: TASK 4: Robustness to Noise
#-----------------

  

#Part: NOise Injection(10% label noise in class) and Implementation 

# -------------------------
# Create a NOISY version of your dataset (credit_clean)
# by Fliping class label for a random 10% of instances
# -------------------------
set.seed(2026)

credit_noise <- credit_clean %>%
  mutate(class = as.factor(class)) %>%
  filter(complete.cases(.))

n_total <- nrow(credit_noise)
n_flip  <- ceiling(0.10 * n_total)

flip_idx <- sample(seq_len(n_total), size = n_flip, replace = FALSE)

# Flip labels: bad <-> good
credit_noise$class[flip_idx] <- factor(
  ifelse(as.character(credit_noise$class[flip_idx]) == "bad", "good", "bad"),
  levels = levels(credit_noise$class)
)

cat("\n--- Noise injected ---\n")
cat("Total rows:", n_total, "\n")
cat("Flipped rows (10%):", n_flip, "\n")
cat("\nClass distribution BEFORE noise (credit_clean):\n")
print(table(credit_clean$class))
cat("\nClass distribution AFTER noise (credit_noise):\n")
print(table(credit_noise$class))

# -------------------------
# preprocessing steps of noisy dataset:
# -------------------------

# Split (70/30 stratified) like you did before
set.seed(123)
idx_train_n <- createDataPartition(credit_noise$class, p = 0.70, list = FALSE)
train_raw_n <- credit_noise[idx_train_n, ]
test_raw_n  <- credit_noise[-idx_train_n, ]

# Convert logical/character to factor
train_raw_n <- train_raw_n %>%
  mutate(across(where(is.logical), ~ factor(.x, levels = c(TRUE, FALSE)))) %>%
  mutate(across(where(is.character), as.factor))

test_raw_n <- test_raw_n %>%
  mutate(across(where(is.logical), ~ factor(.x, levels = c(TRUE, FALSE)))) %>%
  mutate(across(where(is.character), as.factor))

# Integer-coded categorical -> factor (<=10 distinct), excluding class
int_cols_n <- names(train_raw_n)[sapply(train_raw_n, is.integer)]
int_like_cat_n <- int_cols_n[sapply(train_raw_n[int_cols_n], function(x) dplyr::n_distinct(x, na.rm = TRUE) <= 10)]
int_like_cat_n <- setdiff(int_like_cat_n, "class")

if (length(int_like_cat_n) > 0) {
  train_raw_n[int_like_cat_n] <- lapply(train_raw_n[int_like_cat_n], factor)
  test_raw_n[int_like_cat_n]  <- lapply(test_raw_n[int_like_cat_n], factor)
}

# Align factor levels in test to match train
for (col in names(train_raw_n)) {
  if (is.factor(train_raw_n[[col]])) {
    test_raw_n[[col]] <- factor(test_raw_n[[col]], levels = levels(train_raw_n[[col]]))
  }
}

# Drop NA rows (safety)
train_raw_n <- train_raw_n[!apply(is.na(train_raw_n), 1, any), ]
test_raw_n  <- test_raw_n[!apply(is.na(test_raw_n), 1, any), ]

# Feature selection (Chi-square top 15)
N <- 15
weights_n <- chi.squared(class ~ ., train_raw_n)

subset_n <- cutoff.k(weights_n, N)
f_n <- as.simple.formula(subset_n, "class")

selected_features_n <- all.vars(f_n)
selected_features_n <- setdiff(selected_features_n, "class")

train_fs_n <- train_raw_n %>% select(all_of(selected_features_n), class)
test_fs_n  <- test_raw_n  %>% select(all_of(selected_features_n), class)

# Align factor levels after FS
for (col in names(train_fs_n)) {
  if (is.factor(train_fs_n[[col]])) {
    test_fs_n[[col]] <- factor(test_fs_n[[col]], levels = levels(train_fs_n[[col]]))
  }
}

train_fs_n$class <- as.factor(train_fs_n$class)
test_fs_n$class  <- factor(test_fs_n$class, levels = levels(train_fs_n$class))

cat("\n--- Noisy selected-feature formula f_n ---\n")
print(f_n)

# -------------------------
# 3) Running CV procedure of Task 3 (10-fold, 3 repeats)
# -------------------------
trctrl_n <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = "final"
)

set.seed(3333)

dtree_cv_n <- train(
  form = f_n,
  data = train_fs_n,
  method = "rpart",
  parms = list(split = "information"),
  trControl = trctrl_n,
  tuneLength = 5
)

part_cv_n <- train(
  form = f_n,
  data = train_fs_n,
  method = "PART",
  trControl = trctrl_n,
  tuneLength = 5
)

jrip_cv_n <- train(
  form = f_n,
  data = train_fs_n,
  method = "JRip",
  trControl = trctrl_n,
  tuneLength = 5
)


get_fold_acc_from_pred <- function(fit_obj) {
  pred_df <- fit_obj$pred
  acc_by_fold <- pred_df %>%
    group_by(Resample) %>%
    summarise(Accuracy = mean(pred == obs), .groups = "drop") %>%
    arrange(Resample)
  return(acc_by_fold$Accuracy)
}

acc_dt_n   <- get_fold_acc_from_pred(dtree_cv_n)
acc_part_n <- get_fold_acc_from_pred(part_cv_n)
acc_jrip_n <- get_fold_acc_from_pred(jrip_cv_n)

cat("\n--- Noisy data: fold accuracy vector lengths (should be 30) ---\n")
print(c(DT = length(acc_dt_n), PART = length(acc_part_n), JRip = length(acc_jrip_n)))

cat("\n--- Noisy data: mean accuracies ---\n")
print(c(DT = mean(acc_dt_n), PART = mean(acc_part_n), JRip = mean(acc_jrip_n)))

# -------------------------
# 4)  Doing the ANOVA + Tukey on the noisy data
# -------------------------
acc_df_n <- data.frame(
  Accuracy = c(acc_dt_n, acc_part_n, acc_jrip_n),
  Model = factor(rep(c("DecisionTree", "PART", "JRip"), each = 30))
)

anova_fit_n <- aov(Accuracy ~ Model, data = acc_df_n)

cat("\n================= ANOVA TABLE (10% NOISE) =================\n")
anova_out_n <- summary(anova_fit_n)
print(anova_out_n)

p_val_n <- anova_out_n[[1]][["Pr(>F)"]][1]

if (!is.na(p_val_n) && p_val_n < 0.05) {
  cat("\nANOVA: significant difference (p < 0.05). Running Tukey...\n")
  tuk_n <- TukeyHSD(anova_fit_n)
  print(tuk_n)
  
  mean_by_model_n <- acc_df_n %>%
    group_by(Model) %>%
    summarise(mean_accuracy = mean(Accuracy), .groups = "drop") %>%
    arrange(desc(mean_accuracy))
  
  cat("\nMean accuracy by model (10% NOISE):\n")
  print(mean_by_model_n)
  
  cat("\nMost accurate model under 10% noise:\n")
  print(mean_by_model_n$Model[1])
  
} else {
  cat("\nANOVA: NOT significant (p >= 0.05). No difference among classifiers under 10% noise.\n")
}
