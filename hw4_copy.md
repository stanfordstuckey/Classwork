---
output:
  pdf_document: default
  html_document: default
---

# Homework 4: Feature Engineering with Clinical Data [120 points]

### BIOMEDIN 215 (Data Science for Medicine), Fall 2021 

### Due: Tuesday, October 26th, 2021


In this assignment you will gain experience transforming clinical data into sets of features for downstream statistical analysis. You will practice using common time-saving tools in the R programming language that are ideally suited to these tasks.

You will primarily be building off of the cohort that you developed in the Cohort Building homework. In particular, you will extract features from vitals, diagnosis codes, and more that can be used to predict the future development of septic shock. 

You will not be replicating the models presented in ["A targeted real-time early warning score (TREWScore) for septic shock" by Henry et al.](http://stm.sciencemag.org/content/7/299/299ra122.full) directly, but we include a link to the paper for your reference.

All of the data you need for this assignment is available on Canvas.

Please edit this document directly using either Jupyter or R markdown in RStudio and answer each of the questions in-line. Turn in a single .pdf document showing all of your code and output for the entire assignment, with each question clearly demarcated. Submit your completed assignment through Canvas.

## 0. Getting Ready

The first thing we need to do is load all of the packages we will use for this assignment. Please load the packages `tidyverse`, `lubridate`, `data.table`, `Matrix`, and `glmnet`. Also, please run the command `Sys.setenv(TZ='UTC')`.

install.packages("Matrix")
install.packages("glmnet")

library(tidyverse)
library(lubridate)
library(data.table)
library(Matrix)
library(glmnet)

Sys.setenv(TZ='UTC')

## 1. Defining labels for prediction

-----
##### 1.1 (10 pts)

We are going to take the 1000 subject development cohort you worked with on the previous assignment and explore some methods of feature engineering for the task of predicting whether a patient will go on to develop septic shock. You will start with a dataset similar to what you might have generated at the end of the prior assignment. This dataset is available in the file `cohort_labels.csv`.

The prediction problem motivating this assignment is to predict, at 12 hours into an admission, whether septic shock will occur during the remainder of the admission, with at least 3 hours of lead time. Your task is to engineer a set of features that may be used as the inputs to a model that makes this prediction.

We will derive the **labels** and **index times** in a way that aligns with the task description above. Note that this is not the same procedure as in the TREWscore paper.

We will use the following definitions:

* We will only assign labels to admissions of at least twelve hours in duration.
* An admission is assigned a negative label if septic shock does not occur at any time during the admission.
* An admission is assigned a positive label if septic shock occurs fifteen hours after admission or later.
* Admissions where the earliest time of septic shock occurs prior to fifteen hours after admission are removed from the study.
* For admissions that have valid labels, we assign an index time at twelve hours into the admission. For prediction, we only use information that occurs before the index time.
* In the case that a patient has multiple admissions for which a valid index time and label may be assigned, we only use the latest one.

To begin, given the above definitions, load `cohort_labels.csv` and `ADMISSIONS.csv` and derive the binary classification labels for septic shock and the corresponding index times for each patient in the dataframe. The result should be a dataframe with one row per patient and additional columns for `index_time` and `label`.

How many patients receive a positive or negative label?

----

----
## 2. Building a Patient-Feature Matrix for the Septic Shock Cohort

Now that we know have derived labels and index times for each patient in our cohort, we can start to engineer some features from the data that occur prior to the index times and will be useful for predicting onset of septic shock.

### Diagnoses


##### 2.1 (2 pts)

Let's first deal with diagnoses. Load `DIAGNOSES_ICD.csv`. We would like to find the diagnoses that occurred before the index time for each patient, but it looks like there is no time recorded in the diagnosis table.

Which table and columns in MIMIC would you use to find the times of each diagnoses? Justify your response.

Use the online documentation to find out.

----

-----
##### 2.2 (3 pts)

Use the table you have selected in conjunction with the diagnoses and your cohort table to filter the diagnoses for each patient that were recorded before the index time. The final result should have the columns `subject_id`, `hadm_id`, `diagnosis_time`, `icd9_code`, and `index_time`.

How many subjects have diagnoses recorded prior to the index_time? Does the resulting number make sense?

----

-----
##### 2.3 (4 pts)
What are the top 10 most common diagnosis codes (by number of unique patients who had the code in their history) in the data frame resulting from question 2.2? Look up the top 3 codes online and report what they refer to.

----

-----
##### 2.4 (4 pts)

For the set of codes and patients that remain after the index time filtering step, make a histogram demonstrating the distribution of the number of unique diagnostic histories that the codes belong to. In other words, generate a histogram of the count data you generated in 2.3. The x-axis should represent the total number of unique patient histories in which a code appears. The y-axis should represent the total count of codes within each 'number of patient histories' interval.

In 1-2 sentences, interpret the results

----

-----
##### 2.5 (5 pts)
As you observed from the plot you created above, there are many rare diagnoses, resulting in a sparse feature space. One way to manage this is to identify rare (and similarly, very common) features using *Information content (IC)*. IC is a measure of specificity based on the frequency of occurrence of features.

The IC of a feature that occurs in a set of records is calculated as 

$-log_2 \left( \frac{count(\text{feature})}{count(\text{record})} \right)$

Use this equation to calculate the IC of ICD9 codes based on their occurrence in the diagnosis records for the sepsis cohort.

-----
##### 2.6 (3 pts)

What is the range (min and max) of ICs observed in your data? What are the 10 most specific ICD9 codes?

---

-----
##### 2.7 (2 pts)
Filter the set of ICD9 codes for the diagnoses associated with the set of admissions to those with an IC between 4 and 9.

---

-----
##### 2.8 (12 pts)
Now we have our diagnoses features and the times they occured for each patient. All that is left to do is to create a patient-feature matrix that summarizes and organizes the diagnoses features. In this matrix, each row is an patient and each column in a diagnosis code, time binned by whether or not it occured in the preceeding 6 months prior to the index time. In other words, we are going to generate two features for each diagnosis code where one feature represents the count of the number of times the code was observed in the six months prior to the index time and the other features represents the number of times that code was observed in the medical history older than six months. The way to implement this is using a `mutate` to create a time bin indicator and then grouping on that before summarizing and spreading. Use `unite` before spreading to create a unique name for each feature based on its diagnosis code and time bin.

Note that the ICU stay is the first time many patients have been seen at this hospital, so most patients may have few or no prior recorded diagnoses.

What are the dimensions of your resultant dataframe?

----

### Notes

-----
##### 2.9 (4 pts)
Now let's add features from notes. To do so, we'll have to process some text.

The `noteevents` table in MIMIC is too large and unwieldy to load into R, so we've extracted the rows from that table that you will need. The result is in the file `notes_small_cohort_v2.csv`. Load it into R and examine it with `head()`.

Pay attention to the fact that notes are stored in MIMIC with chartdate information, but that charttime is mostly empty. The way that we will account for this in the context of our cutoff times is by filtering the data such that we only consider notes that were recorded prior to the day corresponding to the cutoff time for each patient. Perform the necessary filtering. How many notes are present before and after the filtering step?

---

----
#### 2.10 (2 pts)
UMLS terminologies provide concept hierarchies, as well as sets of terms for individual concepts. For example, there are more than 50 terms in UMLS terminologies for the concept 'myocardial infarction'!

Soon, you will use the SNOMED CT hierarchy and UMLS term sets to construct a dictionary of terms for inflammatory disorders, and then search for those terms in MIMIC III notes.

First, load `snomed_ct_isaclosure.csv` and `snomed_ct_str_cui.csv` in R, and examine them with `head()`. `snomed_ct_isaclosure.csv` contains the child-parent CUI relationships for all of SNOMED CT. `snomed_ct_str_cui.csv` contains the terms (each with a unique term identifier, tid) for each SNOMED CT CUI.

Join `snomedct_isa_closure` with `snomedct_cui_string` to find all terms for each CUI (including the terms associated with its children).

---

-----
##### 2.11 (6 pts)

One feature that is very likely to impact the likelihood of a patient to develop septic shock is whether they currently have or have a history of inflammatory disorders. Let's extract information from clinical notes to look for the presence of this class of disease.

First, use the CUI for inflammatory disorders in SNOMED CT ("C1290884"), and construct a dictionary of all terms (a set of terms) corresponding to inflammatory disorders that have 20 characters or fewer. How many terms are in the dictionary?

---

-----
##### 2.12 (7 pts)

Using any method you like, use your dictionary to determine if the note text contains each of the first fifty terms in your dictionary (limited for computational purposes). Your answer should have the columns `note_id` (the same as `row_id` in `notes.csv`), `subject_id`, `chartdate`, and as many more columns as there are terms in the dictionary (50).

What are the dimensions of the resulting dataframe?

---

-----
##### 2.13 (6 pts)

Now use your result from question 2.10 to normalize terms back to their concepts and construct a dataframe of `subject_id`, `chartdate` and `concept`.

---

-----
##### 2.14 (7 pts)

As with the diagnoses, we must transform this data into a patient-feature matrix. Use `dplyr` and `tidyr` to transform this table of concept mentions into a patient-feature matrix where each row is a patient and each column is the presence or absence of a concept. Do not do any time-binning. Each concept should have only one column. Instead of counts, use a binary indicator to indicate that the concept was present in the patient's notes prior to the cutoff time.

What are the dimensions of the resulting table?

----

### Vitals

-----
##### 2.15 (2 pts)

Now let's engineer some features from vital sign measurements also relevant to predicting septic shock.

Here we will work with the patient's heart rates. Load the file `vitals_small_cohort.csv` (this file will be familiar to you at this point). Once you have done so, filter measurements so that you are only looking at Heart Rate  measurements that occured prior to the cutoff time for the set of patients in our cohort.

How many patients are left in the dataframe after performing this filtering step?

----

-----
##### 2.16 (4pts)

One feature of interest might be the latest value of the heart rate before the cutoff time. Use `dplyr` to make a dataframe with three columns: `subject_id`, `latest_heart_rate`, and `charttime`. 

What is the average value of the latest recorded heart rate in this cohort? Additionally, make a histogram or density plot of the latest heart rate colored by whether a patient develops septic shock.

----

-----
##### 2.17 (4 pts)

The latest recorded heart rate might not be a useful feature to use if the latest recording is not near the index time. Make a density plot of the time difference between the latest heart rate recording and the cutoff time colored by whether a patient develops septic shock. Feel free to modify the axes limits if that helps you interpret the plot.

----

-----
##### 2.18 (5 pts)
Some patients might have many heart rate recordings, and only using the last one might not be the best idea- it's possible the latest measurement is an outlier. Let's try to leverage all the heart rate measurements we have by creating a time-weighted average heart rate. Use the formula $w = e^{(-|\Delta t| - 1)}$ to calculate the weights of each measurement, where $\Delta t$ is the time difference between the measurement time and the cutoff time in hours. Calculate the weighted average with the formula $\bar{x}_w = \sum(x_i w_i)/\sum(w_i)$. The result should be a dataframe with two columns: `subject_id` and `time_wt_avg`.

What is the average time-weighted average heart rate across all patients? 

----

-----
##### 2.19 (4 pts)
Let's do a sanity check to see if what we've done makes sense. We expect that the time-weighted average heart rate and the latest recorded heart rate should be similar.

Make a scatterplot of the latest recorded heart rate (x-axis) and the time-weighted average heart rate (y-axis) of each patient.

----

### Stitching together Disease, Text and Vitals Features

-----
##### 2.20 (4 pts)
Our final patient-feature matrix will simply be the amalgamation of the different feature matrices we've created. Use an outer join to combine the columns of the feature matrices from diagnoses, notes, and heart rate measurements. Not all patients have diagnoses or note features, so fill in any NA values with 0 to indicate that there were no diagnoses or notes counted. Similarily, not all subjects have heart rate measurements.  Fill null values for these features with a simple column mean imputation. Use `names` to look at all the features and make sure everything seems ok.


How many total features are there?

----

## 3. Open ended feature engineering - Do something cool! (20 points)

Having made it this far, you have picked up a few generalizable techniques that can now be used to extract features from various modalities of clinical data. To test the skills you've learned thus far, you now have free reign to get creative and derive whatever additional features you would like and use them alongside the disease, text and vitals features as input to a simple classifier. To help you with your task, we provide you with CSV files for ALL of the tables in MIMIC III where each table has been filtered to contain only the records for the patients in our small cohort. These are stored in the folder `additional_data`. To start, we provide you with some baseline code below that runs a logistic regression classifier with a Lasso L1 penalty and reports a cross-validation AUC-ROC.

More concretely, do the following:
* Outside of the features we engineered previously in the assignment, derive additional features that utilize at least five of the additional data tables. You may use tables that we have previously worked with as a part of the assignment, but we encourage you to explore these new data sources. Caveats: definition tables (e.g. d_items) do not count towards the five and using any combination of chartevents tables counts as a single table.
* Combine your derived features into a patient-feature matrix
* Adapt the model-fitting code provided below to your new dataset
* Write 1-2 paragraphs discussing what and how many features you derived. Additionally, discuss the effects of those features on the performance of the classifier.

Feel free to modify anything you would like in the code below to fufill your purposes. That said, you are not being evaluating the performance of your classifier and are instead being evaluated on your feature engineering procedure and discussion, so do not expend too much effort in getting a good AUC-ROC.


```R
# Baseline implementation - provided

library(caret)

# Requires caret package
# Assumes that feature matrix from 2.22 is stored in X
# Constructs a label vector from label_df - a dataframe containing the septic_shock labels

# Pipeline automatically performs median imputation, centering, scaling, and near-zero-variance feature filtering

# Runs the data through a logistic regression with a lasso penalty
# Reports area under the receiver operating curve as performance metric

baseline_df <- X %>% left_join(select(label_df, subject_id, label)) #%>% mutate()
baseline_features <- baseline_df %>% select(-label, -subject_id) %>% as.data.frame()
labels <- baseline_df %>% select(label) %$% relevel(factor(as.numeric(label), 
                                                                 labels = c('normal', 
                                                                            'septic_shock')), 'septic_shock')
config <- trainControl(method="cv", 
                       number=5, 
                       returnResamp="all",
                       classProbs=TRUE, 
                       summaryFunction=twoClassSummary,
                       verboseIter = TRUE)

baseline_model <- train(baseline_features, 
                             labels, 
                             preProcess = c('medianImpute', 'center', 'scale', 'nzv'),
                             method = "glmnet", 
                             trControl = config,
                             metric = "ROC",
                             tuneGrid = expand.grid(alpha = 1,
                                                    lambda = seq(0.001, 0.1, by = 0.001))
                       )

print(baseline_model$results)
```

----

### Done!

That's it! You've gone through the major steps of transforming different kinds of data stored in a longitudinal database into a patient-feature matrix that we can use for association tests and prediction tasks. Along the way we hope you have gained practice in how to effectively use the `dplyr` and `tidyr` packages to manipulate data and the `ggplot2` package to make visual diagnostics. You are well on your way to being able to perform a clinical informatics study.


```R

```

## Feedback (0 points)
#####  How much time did you spend on this assignment?

#####  How much did you learn? Choose one (type your answer after the table):
   A | B | C | D | E |
   --|---|---|---|---|
   a great deal |  a lot  |  a moderate amount | a little | none at all|

##### Did you do any of the following: go to office hours, post on canvas, e-mail TAs? If so, which?
