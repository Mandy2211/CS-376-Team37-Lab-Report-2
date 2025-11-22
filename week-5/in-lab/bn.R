library(bnlearn)
library(e1071)
library(caret)

options(repos = c(CRAN = "https://cran.r-project.org"))
install.packages("bnlearn")
install.packages("e1071")
install.packages("caret")
install.packages("gRain")


library(bnlearn)
library(e1071)
library(caret)

course.grades <- read.table("2020_bn_nb_data.txt",header = TRUE)

course.grades[] <- lapply(course.grades, function(x) {
    if (is.character(x)) {
        as.factor(x)
    } else {
        x
    }
})

if (!"QP" %in% colnames(course.grades)) {
    stop("The 'QP' variable does not exist in the dataset.")
}

course.grades.net <- hc(course.grades)
plot(course.grades.net)
course.grades.net.fit <- bn.fit(course.grades.net, course.grades)

print(nodes(course.grades.net))

input_grades <- data.frame(EC100 = factor("DD", levels = levels(course.grades$EC100)),
                           IT101 = factor("CC", levels = levels(course.grades$IT101)),
                           MA101 = factor("CD", levels = levels(course.grades$MA101)))

predicted_ph100 <- predict(course.grades.net.fit, node = "PH100", data = input_grades, method = "exact")
cat("Predicted grade in PH100 for EC100: DD, IT101: CC, MA101: CD is:", predicted_ph100, "\n")

set.seed(123)
n_trials <- 20
accuracy_results <- numeric(n_trials)

for (i in 1:n_trials) {
    train_index <- sample(1:nrow(course.grades), size = 0.7 * nrow(course.grades))
    train_data <- course.grades[train_index, ]
    test_data <- course.grades[-train_index, ]
    
    nb_classifier <- naiveBayes(QP ~ ., data = train_data)
    predictions <- predict(nb_classifier, newdata = test_data)
    cm <- confusionMatrix(predictions, test_data$QP)
    accuracy_results[i] <- cm$overall['Accuracy']
}

mean_accuracy_nb <- mean(accuracy_results, na.rm = TRUE)
cat("Mean accuracy of Naive Bayes classifier over 20 trials:", mean_accuracy_nb, "\n")

bayes_accuracy_results <- numeric(n_trials)

for (i in 1:n_trials) {
    train_index <- sample(1:nrow(course.grades), size = 0.7 * nrow(course.grades))
    train_data <- course.grades[train_index, ]
    test_data <- course.grades[-train_index, ]

    bayes_net <- hc(train_data)
    bayes_net_fit <- bn.fit(bayes_net, train_data)
    bayes_predictions <- predict(bayes_net_fit, node = "QP", data = test_data, method = "exact")
    cm_bayes <- confusionMatrix(bayes_predictions, test_data$QP)
    bayes_accuracy_results[i] <- cm_bayes$overall['Accuracy']
}

mean_accuracy_bn <- mean(bayes_accuracy_results, na.rm = TRUE)
cat("Mean accuracy of Bayesian Network classifier over 20 trials:", mean_accuracy_bn, "\n")