Abstract
This report presents a detailed exploration of a machine learning project focused on classifying text messages as spam or ham (not spam). The project leverages various Python libraries, including NumPy, pandas, scikit-learn, NLTK, Matplotlib, and Seaborn, to perform data loading, preprocessing, feature extraction, model training, evaluation, and visualization. Two advanced ensemble learning techniques, Voting Classifier and Stacked Generalization, are implemented to enhance model performance.
1. Introduction
The goal of this project is to develop an accurate and reliable machine learning model to classify text messages as spam or non-spam. This is crucial for applications in email filtering, SMS services, and other communication platforms to improve user experience and security.
2. Libraries and Tools
The project uses the following libraries and tools:
- NumPy: For numerical calculations and handling multi-dimensional arrays.
- pandas: For data manipulation and analysis.
- scikit-learn: For machine learning algorithms and model evaluation.
- NLTK: For natural language processing tasks.
- Matplotlib and Seaborn: For data visualization.
3. Data Loading
The dataset is loaded using pandas. The typical structure of the data includes features like message content and target labels (spam or ham).
Example code for data loading
```
import pandas as pd
data = pd.read_csv('path_to_your_dataset.csv')
data.sample(5)
```
4. Data Preprocessing
Preprocessing steps include:
- Converting text to lowercase.
- Tokenization.
- Removing stop words and punctuation.
- Stemming.
Example code for text preprocessing:
```
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)
```
5. Feature Extraction
Text data is transformed into numerical features using methods such as TF-IDF (Term Frequency-Inverse Document Frequency).
Example code for feature extraction
```
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']
```
6. Model Training
6.1 Individual Models
Various machine learning models are trained and evaluated, including:
- Naive Bayes
- Support Vector Machines (SVM)
- Random Forest

Example code for training an individual model:
```
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
```
6.2 Voting Classifier
A Voting Classifier combines predictions from multiple base models to improve overall performance.

Example code for a Voting Classifier:
```
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

svm_model = SVC(kernel='linear', probability=True)
rf_model = RandomForestClassifier(n_estimators=100)

voting = VotingClassifier(estimators=[
    ('nb', nb_model),
    ('svm', svm_model),
    ('rf', rf_model)
], voting='hard')

voting.fit(X_train, y_train)
```

6.3 Stacked Generalization
Stacked generalization (stacking) is an advanced ensemble technique that combines predictions from multiple base models to form a new dataset, which is then used to train a meta-model.

Example code for stacking:
```
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Define base models
base_models = [
('nb', nb_model),
('svm', svm_model),
('rf', rf_model)
]

# Define the meta-model
meta_model = LogisticRegression()

# Create the stacking classifier
stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the stacking classifier
stacking.fit(X_train, y_train)
```
7. Model Evaluation
Models are evaluated using metrics such as accuracy and precision. Visualization of performance metrics is done using Seaborn and Matplotlib.

Example code for model evaluation:
```
from sklearn.metrics import accuracy_score, precision_score

# Evaluate Voting Classifier
y_pred_voting = voting.predict(X_test)
print("Voting Classifier - Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Voting Classifier - Precision:", precision_score(y_test, y_pred_voting))

# Evaluate Stacking Classifier
y_pred_stacking = stacking.predict(X_test)
print("Stacking Classifier - Accuracy:", accuracy_score(y_test, y_pred_stacking))
print("Stacking Classifier - Precision:", precision_score(y_test, y_pred_stacking))
```
8. Results
Results are visualized using various plots such as heatmaps, pie charts, and bar plots to represent the performance of models and the distribution of data.

Example visualization of word cloud for spam and ham messages:
```
import matplotlib.pyplot as plt
from wordcloud import WordCloud

spam_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black').generate(data[data['label']=='spam']['message'].str.cat(sep=" "))
plt.imshow(spam_wc)
plt.axis('off')
plt.show()

ham_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(data[data['label']=='ham']['message'].str.cat(sep=" "))
plt.imshow(ham_wc)
plt.axis('off')
plt.show()
```
9. Conclusion
The project successfully demonstrates the use of machine learning and natural language processing techniques to classify text messages. The results indicate that the implemented models, including the Voting Classifier and Stacked Generalization, perform effectively. Stacking, in particular, shows a significant improvement in performance, leveraging the strengths of multiple base models.
10. Future Work
Future improvements could include:
- Experimenting with more advanced NLP techniques like word embeddings.
- Incorporating additional features like metadata of messages.
- Hyperparameter tuning for better model performance.
This detailed report outlines the process and findings of the spam classification project. The combination of preprocessing, feature extraction, model training, evaluation, and advanced ensemble techniques such as Voting Classifier and Stacked Generalization provides a robust framework for text classification tasks.
