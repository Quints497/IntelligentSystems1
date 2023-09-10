# IntelligentSystems1


# Spam Filter

This project aims to develop a robust spam filter using machine learning techniques to automatically categorise emails as spam or non-spam. The filter helps users avoid the hassle of dealing with unimportant or scam emails, ensuring that important messages are not missed.

## Dataset

The dataset used for training the spam filter consists of 4601 instances, with each instance containing 57 attributes. The attributes include features such as word frequency, character frequency, and capital run length statistics. The target attribute represents whether an email is spam or not, with a binary value (0 or 1)

## Algorithm

The selected algorithm for this project is a hard voting classifier, which combines a Neural Network and Random Forest classifier. Through extensive experimentation and hyperparameter tuning using Grid Search with 5-fold cross-validation, these two classifiers have shown the highest performance individually. By leveraging the strengths of both algorithms, we aim to achieve higher accuracy and precision in spam classification.

## Evaluation

The performance of the spam filter is evaluated using two key metrics: accuracy and precision. To establish a benchmark for comparison, the K-Nearest Neighbours algorithm has been selected. By comparing the results of the voting classifier with the benchmark, we can assess the effectiveness of our approach.

## Results

The performance of the voting classifier significantly outperforms the benchmark algorithm, as demonstrated in the following graphs:

<h6>Accuracy Over 5 Tests</h6>
<img width="361" alt="image" src="https://github.com/Quints497/IntelligentSystems1/assets/70848538/88a9d48e-712f-4535-9d37-99820b15a278">

<h6>Precision Over 5 Tests</h6>
<img width="360" alt="image" src="https://github.com/Quints497/IntelligentSystems1/assets/70848538/a450b513-65cc-4db1-9dbf-bc682891aa65">

<h6>Average Accuracy</h6>
<img width="360" alt="image" src="https://github.com/Quints497/IntelligentSystems1/assets/70848538/884a8fd8-7130-430a-a4a2-378a20b610ae">

<h6>Average Precision</h6>
<img width="361" alt="image" src="https://github.com/Quints497/IntelligentSystems1/assets/70848538/c0e5a6fc-831f-40a9-85e1-6b3d2ca97a7d">

## Usage

To use the spam filter, follow these steps:
<ol>  
  <li> Install the required dependencies (Python 3, pandas, scikit-learn)</li>
  <li> Download and preprocess the dataset</li>
  <li> Train the spam filter using the provided code and the selected algorithm</li>
  <li> Evaluate the performance using the provided evaluation metrics</li>
  <li> Integrate the spam filter into your email system for automatic classification</li>
</ol>

## Conclusion

The spam filter developed in this project offers an efficient solution for automatically identifying and filtering out spam emails. By combining a Neural Network and Random Forest classifier in a hard voting approach, I have achieved superior performance compared to the benchmark algorithm. 

## License

This project is licensed under the MIT License
