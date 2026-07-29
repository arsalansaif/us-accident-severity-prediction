# US Accident Severity Prediction

A data-mining project that explores the **US-Accidents** dataset and builds machine-learning models to predict the **severity** of a traffic accident from its contextual features (location, time, weather, and road attributes).

## What's inside

**Exploratory analysis**
- Accident counts by state (top-10 ranking)
- Most common words in the descriptions of the most severe (severity-4) accidents
- Frequency of road features (junction, crossing, traffic signal, stop, etc.)
- Accident distribution across weekdays and time

**Modeling**
Severity is treated as a classification target. Several classifiers from scikit-learn are trained and compared, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Naive Bayes (Bernoulli / Multinomial / Gaussian)

Feature handling includes categorical encoding (`category_encoders`), scaling (`MinMaxScaler`), and text stop-word removal for description fields.

## Repository structure

| File | Purpose |
|---|---|
| `code.py` | Full pipeline: EDA, preprocessing, model training and evaluation |
| `report.pdf` | Project write-up with results |

## Requirements

Python 3, `pandas`, `numpy`, `scikit-learn`, `category_encoders`, `nltk`, `matplotlib`, `seaborn`.

## Data

Uses the public **US-Accidents (March 2023)** dataset. Update the CSV path at the top of `code.py` to point to your local copy.
