# Credit Risk Classification and Rule Induction

**Timeline:** April 2026  
**Category:** Main project

This project compares interpretable credit-risk classifiers and extracts human-readable decision rules. The workflow covers exploratory analysis, feature selection, tree-based classification, PART rules, and RIPPER/JRip induction with dimensionality-reduction visual diagnostics.

## Stack

R, tidyverse, caret, rpart, RWeka, FSelector, Rtsne, ggplot2, decision trees, PART, and RIPPER/JRip.

## Repository contents

- `src/credit_risk_project.R` — analysis and modeling workflow.
- `rules/` — exported rule sets from the interpretable learners.
- `reports/` — milestone reports.
- `data/README.md` — provenance and excluded-data instructions.

Run the script in an R environment after placing an authorized copy of the credit-risk data at the documented local path.
