# Titanic Survival Prediction using Random Forest Classifier

## Project Overview
This project focuses on predicting passenger survival on the **Titanic dataset** using a **Random Forest Classifier**. It demonstrates a complete machine learning pipeline including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization of individual decision trees within the random forest ensemble.

The project is designed for learning and demonstration purposes, showcasing how ensemble learning improves prediction performance compared to a single decision tree.

---

## Objectives
- Analyze factors affecting passenger survival on the Titanic
- Perform exploratory data analysis using visualizations
- Preprocess categorical and missing data
- Train a Random Forest classification model
- Evaluate model performance using standard metrics
- Visualize individual trees from the Random Forest

---

## Technologies Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## Dataset Description
The dataset (`titanic.csv`) contains passenger information with the following relevant features:

### Features Used
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare

### Target Variable
- **Survived**: Survival status (0 = Did not survive, 1 = Survived)

---

## Data Preprocessing
- Removed rows with missing target values
- Selected relevant numerical and categorical features
- Encoded the `Sex` feature:
  - Female → 0
  - Male → 1
- Filled missing age values using the median age
- Split dataset into training and testing sets (80/20 split)

---

## Exploratory Data Analysis (EDA)
Count plots were generated to analyze survival trends across:
- Passenger class
- Gender
- Age
- Number of siblings/spouses
- Number of parents/children
- Fare distribution

These visualizations help identify key survival patterns before model training.

---

## Model Description
### Random Forest Classifier
- Ensemble learning technique using multiple decision trees
- Reduces overfitting compared to a single decision tree
- Aggregates predictions from multiple estimators

The model was trained using default hyperparameters from scikit-learn’s `RandomForestClassifier`.

---

## Model Training and Testing
- Training data used to fit the Random Forest model
- Predictions made on the test dataset
- Performance evaluated using:
  - Accuracy Score
  - Precision, Recall, and F1-Score (Classification Report)

---

## Model Evaluation
The trained model outputs:
- Overall accuracy of predictions
- Detailed classification report showing class-wise performance

These metrics provide insight into how well the model distinguishes between survivors and non-survivors.

---

## Visualization of Random Forest Trees
To improve interpretability:
- Multiple individual decision trees from the Random Forest ensemble were visualized
- Trees at different indices were plotted to show model diversity
- Each tree displays:
  - Feature splits
  - Decision thresholds
  - Class predictions
  - Node impurity and sample distribution

This helps understand how ensemble models learn different decision boundaries.

---

## How to Run the Project

### Prerequisites
Ensure the following libraries are installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Steps
1. Place `titanic.csv` in the specified directory or update the file path
2. Run the Python script
3. Observe:
   - EDA visualizations
   - Printed accuracy and classification report
   - Decision tree visualizations from the Random Forest

---

## Learning Outcomes
- Practical understanding of ensemble learning
- Hands-on experience with Random Forest classifiers
- Improved skills in data preprocessing and visualization
- Ability to interpret individual estimators within an ensemble model

---

## Future Enhancements
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Cross-validation for robust evaluation
- Handling categorical features using advanced encoders
- Deployment as a web-based prediction app

---

## Use Case
This project is suitable for:
- Machine Learning coursework
- Data Science portfolios
- Interview preparation
- Understanding ensemble models in practice

---

## Author
Soban Saeed
Developed as an educational machine learning project to demonstrate Titanic survival prediction using Random Forest classification.
