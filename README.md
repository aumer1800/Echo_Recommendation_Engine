# Echo Recommendation Engine

## Project Overview

**Echo Recommendation Engine** is a movie recommendation system based on **Collaborative Filtering** and **Matrix Factorization**.

The project uses the **MovieLens 1M dataset** to learn the relationship between users and movies from their previous ratings. The system predicts how a user may rate movies they have not rated before and recommends the **Top 5 unseen movies** with the highest predicted ratings.

The main goal of this project is to build a recommendation system that can:

* Analyze user-movie interactions
* Create a user-item rating matrix
* Apply Matrix Factorization using SVD
* Predict ratings for unseen movies
* Recommend the Top 5 movies for a specific user
* Evaluate the recommendation model using RMSE and MAE
* Generate recommendations for different users
* Save and reload the trained model for later use

---

## Project Objectives

The main objectives of this project are:

1. Load and clean the MovieLens dataset.
2. Explore user, movie, and rating information.
3. Create a sparse user-item matrix.
4. Apply Collaborative Filtering.
5. Use **Singular Value Decomposition (SVD)** for Matrix Factorization.
6. Train the recommendation model using existing ratings.
7. Tune important model parameters.
8. Evaluate the model on a test dataset.
9. Create a function that accepts a User ID.
10. Return the Top 5 movies that the user has not rated yet.
11. Test the recommendation system on three different users.
12. Save the trained model so it can be used again without retraining.

---

## Dataset

This project uses the **MovieLens 1M dataset**.

The dataset contains information about:

* Users
* Movies
* Movie genres
* User ratings
* Rating timestamps

### Dataset Files

The project uses three main files:

```text
data/
├── ratings.dat
├── movies.dat
└── users.dat
```

### `ratings.dat`

Contains user ratings for movies.

Main columns:

* `userId` - Unique ID of the user
* `movieId` - Unique ID of the movie
* `rating` - Rating given by the user
* `timestamp` - Time when the rating was created

### `movies.dat`

Contains movie information.

Main columns:

* `movieId` - Unique movie ID
* `title` - Movie title
* `genres` - Movie genres

### `users.dat`

Contains user information.

Main columns:

* `userId` - Unique user ID
* `gender` - User gender
* `age` - User age group
* `occupation` - User occupation
* `zipCode` - User ZIP code

---

## Technologies Used

The project was developed using Python and Jupyter Notebook.

### Tools and Libraries

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Surprise
* Scikit-learn
* Joblib

### Main Machine Learning Algorithm

**Singular Value Decomposition (SVD)**

SVD is used for Matrix Factorization in the Collaborative Filtering system.

---

# Project Workflow

The complete project follows this workflow:

```text
MovieLens Dataset
       ↓
Data Loading
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
User-Item Matrix
       ↓
Train/Test Split
       ↓
SVD Matrix Factorization
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Predict Unseen Movies
       ↓
Top 5 Recommendations
       ↓
Save Trained Model
```

---

# 1. Data Loading

The first step is loading the MovieLens files using Pandas.

The `.dat` files use `::` as the separator, so the Python `read_csv()` function is configured to read this format.

The data is loaded into three DataFrames:

* Ratings
* Movies
* Users

The data is then used for cleaning, analysis, model training, and recommendations.

---

# 2. Data Cleaning

Before training the recommendation model, the data was checked and cleaned.

### Ratings Cleaning

The ratings dataset was checked for:

* Missing values
* Duplicate records
* Invalid User IDs
* Invalid Movie IDs
* Ratings outside the expected range

Only ratings between **1 and 5** were accepted.

The cleaned ratings data was saved as:

```text
ratings_cleaned.csv
```

### Movies Cleaning

The movies dataset was also cleaned.

The following checks were performed:

* Missing movie IDs
* Missing movie titles
* Missing genres
* Duplicate movies

Missing genres were handled using:

```text
Unknown
```

The cleaned movie data was saved as:

```text
movies_cleaned.csv
```

---

# 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset before building the recommendation model.

Several aspects of the dataset were analyzed.

### Rating Distribution

The distribution of ratings was analyzed to understand how users rate movies.

This helps identify whether users mostly give low, medium, or high ratings.

### Most Rated Movies

The Top 10 most-rated movies were identified.

This shows which movies received the highest number of user interactions.

### Highest Rated Movies

Movies with high average ratings were analyzed.

A minimum rating count was used so that movies with only a very small number of ratings would not dominate the results.

### Ratings Per User

The number of ratings given by each user was analyzed.

This helps understand user activity in the dataset.

### Ratings Per Movie

The number of ratings received by each movie was also analyzed.

This helps identify popular movies.

### Genre Analysis

Movie genres were analyzed to understand the types of movies available in the dataset.

---

# 4. Sparsity Analysis

A recommendation dataset normally contains many possible user-item combinations, but users only interact with a small number of items.

For example:

```text
User 1 → Movie 1 → Rated
User 1 → Movie 2 → Not Rated
User 1 → Movie 3 → Not Rated
User 1 → Movie 4 → Rated
```

This creates a **sparse matrix**.

The project calculates the sparsity of the user-item matrix to understand how many possible user-movie interactions are actually available.

---

# 5. User-Item Matrix

The main structure used in Collaborative Filtering is the **User-Item Matrix**.

The rows represent users and the columns represent movies.

Example:

```text
             Movie 1   Movie 2   Movie 3   Movie 4
User 1          5         -         3         -
User 2          -         4         -         5
User 3          2         -         5         -
```

Here:

* Rows = Users
* Columns = Movies
* Values = Ratings
* `-` = No rating/interactions

The project creates this matrix using:

```python
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)
```

Missing values are **not replaced with zero**, because zero would incorrectly mean that the user gave a rating of zero.

Instead, missing values represent that the user has not rated that movie.

---

# 6. Collaborative Filtering

The project uses **Collaborative Filtering**.

Collaborative Filtering makes recommendations using information from user-item interactions.

The basic idea is:

> Users who have similar rating patterns may have similar movie preferences.

For example, if two users both give high ratings to many of the same movies, the system can use this information to predict movies that one user may like.

This project uses **model-based Collaborative Filtering** through Matrix Factorization.

---

# 7. Matrix Factorization

Matrix Factorization is used to represent users and movies using smaller sets of hidden features called **latent factors**.

Instead of working directly with a large user-item matrix, the model learns:

* User latent factors
* Movie latent factors

The idea can be represented as:

```text
User-Movie Rating Matrix
          ↓
   Matrix Factorization
          ↓
 User Factors × Movie Factors
          ↓
 Predicted Ratings
```

These latent factors capture hidden patterns in user preferences and movie characteristics.

For example, the model may learn hidden patterns related to:

* Action preference
* Comedy preference
* Drama preference
* Preference for older/newer movies
* General rating behavior

The model learns these patterns from the ratings rather than manually defining them.

---

# 8. SVD Model

The project uses **Singular Value Decomposition (SVD)** from the Surprise library.

SVD is a Matrix Factorization algorithm commonly used for rating-based recommendation systems.

The model learns from the training ratings and predicts a rating for a user and movie combination.

For example:

```text
User ID: 1
Movie ID: 100

Predicted Rating: 4.35
```

The model can repeat this prediction for movies that the user has not rated.

---

# 9. Train/Test Split

The available ratings were divided into training and testing data.

The project uses:

```text
80% → Training Data
20% → Testing Data
```

The training data is used to learn the user and movie patterns.

The test data is kept separate and is used to measure how well the trained model predicts ratings it has not seen during training.

A fixed random state of `42` was used to make the split reproducible.

---

# 10. Initial SVD Model

An initial SVD model was trained using parameters such as:

```text
n_factors = 100
n_epochs = 20
learning rate = 0.005
regularization = 0.02
```

The model was trained using the training dataset.

---

# 11. Hyperparameter Tuning

Different SVD configurations were tested to find a better model configuration.

The project tested different values of:

### Number of Factors

```text
50
100
150
200
```

### Number of Epochs

```text
10
20
30
```

For each configuration, the model was evaluated using:

* RMSE
* MAE

The results were compared and the configuration with the lowest RMSE was selected.

The final model configuration used in the project is:

```text
n_factors = 50
n_epochs = 20
learning rate = 0.005
regularization = 0.02
```

---

# 12. Model Evaluation

The final model was evaluated using the test dataset.

Two evaluation metrics were used.

## RMSE

**Root Mean Squared Error (RMSE)** measures the difference between actual ratings and predicted ratings.

A lower RMSE means that the predicted ratings are closer to the actual ratings.

The final model achieved:

```text
RMSE = 0.8706
```

---

## MAE

**Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted ratings.

A lower MAE means that the predictions are closer to the actual ratings.

The final model achieved:

```text
MAE = 0.6839
```

### Final Evaluation

| Metric | Result |
| ------ | -----: |
| RMSE   | 0.8706 |
| MAE    | 0.6839 |

These results show the prediction error of the SVD rating model on the test set.

---

# 13. Recommendation Function

A main part of the project is the recommendation function.

The function accepts a specific **User ID**:

```python
get_top_5_recommendations(user_id, model, ratings, movies)
```

The function performs the following steps:

```text
User ID
   ↓
Find movies already rated by user
   ↓
Find movies not rated by user
   ↓
Predict ratings for unseen movies
   ↓
Sort by predicted rating
   ↓
Select Top 5
   ↓
Return movie details
```

The function returns:

* Movie ID
* Movie title
* Movie genres
* Predicted rating

---

# 14. Avoiding Already Rated Movies

The recommendation system does not recommend movies that the user has already rated.

For each user:

1. The system finds the movies already rated by that user.
2. It removes those movies from the recommendation candidates.
3. It predicts ratings only for unseen movies.
4. It selects the five movies with the highest predicted ratings.

This makes the recommendations more useful because the system focuses on movies that are new to the user.

---

# 15. Top 5 Recommendation Example

The recommendation system can be used like this:

```python
get_top_5_recommendations(
    user_id=1,
    model=final_model,
    ratings=ratings,
    movies=movies
)
```

The output contains the five unseen movies with the highest predicted ratings for that user.

Example output format:

```text
movieId | title | genres | predicted_rating
```

The actual recommended movies are generated by the trained model.

---

# 16. Testing Multiple Users

The recommendation function was tested with three different User IDs:

```text
User 1
User 1841
User 5000
```

For each user, the system generated five recommendations.

This was done to demonstrate that the recommendation function works for different users rather than only one user.

The sample recommendations were also saved as:

```text
sample_recommendations.csv
```

---

# 17. Model Saving

After training the final SVD model, it was saved using Joblib.

The saved model file is:

```text
svd_recommendation_model.pkl
```

Saving the model means it does not have to be trained again every time the recommendation system is used.

The model can be loaded later and used to generate recommendations.

---

# 18. Recommendation Script

A separate Python script named:

```text
recommendation.py
```

was created to demonstrate how the trained model can be used outside the notebook.

The script:

1. Loads the trained SVD model.
2. Loads the cleaned ratings data.
3. Loads the cleaned movie data.
4. Uses the recommendation function.
5. Generates Top 5 recommendations.
6. Tests the system with three different users.

This provides a simple example of using the trained recommendation engine after model training.

---

# Project Structure

The project can be organized as follows:

```text
Echo-Recommendation-Engine/
│
├── data/
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
│
├── notebook/
│   └── Echo_Recommendation_Engine.ipynb
│
├── ratings_cleaned.csv
├── movies_cleaned.csv
├── sample_recommendations.csv
├── svd_recommendation_model.pkl
├── recommendation.py
├── requirements.txt
└── README.md
```

---

# How to Run the Project

## 1. Clone the Repository

Clone the GitHub repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Echo-Recommendation-Engine
```

---

## 2. Install Required Libraries

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, the main libraries required are:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-surprise scikit-learn joblib jupyter
```

---

## 3. Open the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

---

## 4. Run the Recommendation Script

After the model and cleaned files have been generated, run:

```bash
python recommendation.py
```

The script will generate Top 5 movie recommendations for the sample users.

---

# Example Usage

The recommendation function can be used for any User ID available in the dataset.

```python
recommendations = get_top_5_recommendations(
    user_id=1,
    model=final_model,
    ratings=ratings,
    movies=movies
)

print(recommendations)
```

The output contains:

```text
Movie ID
Movie Title
Genres
Predicted Rating
```

---

# What I Achieved

Through this project, I built a complete **Collaborative Filtering movie recommendation system** using Matrix Factorization.

The project achieved the following:

* Loaded and processed the MovieLens 1M dataset.
* Cleaned ratings and movie data.
* Performed Exploratory Data Analysis.
* Analyzed rating and user/movie activity.
* Created a sparse user-item matrix.
* Implemented Collaborative Filtering.
* Applied SVD Matrix Factorization.
* Split the data into training and testing sets.
* Performed SVD hyperparameter tuning.
* Evaluated the model using RMSE and MAE.
* Achieved **0.8706 RMSE** and **0.6839 MAE** on the test set.
* Created a User ID based recommendation function.
* Generated Top 5 unseen movie recommendations.
* Tested recommendations on three different users.
* Verified that previously rated movies are excluded.
* Saved the trained model using Joblib.
* Created a separate Python script for using the trained recommendation model.

---

# Use of the Project

A recommendation system like this can be used in many real-world applications.

### Movie Streaming Platforms

A movie platform can recommend movies based on a user's previous ratings and interactions.

### E-Commerce

The same idea can be applied to products.

For example:

```text
User → Products → Ratings/Purchases
```

The system can recommend products that the user has not purchased before.

### Music Platforms

The system can recommend songs or artists based on previous user interactions.

### Book Recommendation

The same approach can recommend books based on user ratings.

### Online Learning Platforms

It can recommend courses or learning materials based on a user's previous activity.

---

# Limitations

Although the project provides a working recommendation system, there are some limitations.

### Rating-Based Recommendations

The current system predicts ratings for unseen movies. It does not directly predict the probability of a user's next click, watch, or purchase.

### Cold Start Problem

New users with very few ratings may be difficult to recommend for because the model has limited information about their preferences.

New movies with very few ratings can also be difficult to recommend.

### Dataset

The model is trained using the MovieLens dataset, so its recommendations are limited to the movies available in that dataset.

### Prediction Cost

The current recommendation function checks unseen movies and predicts their ratings before selecting the Top 5. For a very large production system, more efficient candidate-generation methods would normally be used.

---

# Future Improvements

The project can be improved in several ways:

* Add more recommendation algorithms.
* Compare SVD with other Collaborative Filtering methods.
* Add content-based filtering.
* Build a hybrid recommendation system.
* Improve recommendations for new users.
* Add a web interface.
* Allow users to rate movies through the application.
* Store user interactions in a database.
* Add real-time recommendations.
* Use implicit feedback such as clicks, views, and watch history.
* Deploy the recommendation system as an API.
* Improve recommendation speed for large datasets.

---

# Key Concepts Learned

This project provided practical experience with:

* Data preprocessing
* Data cleaning
* Exploratory Data Analysis
* Sparse matrices
* User-item matrices
* Collaborative Filtering
* Matrix Factorization
* Singular Value Decomposition
* Train/test splitting
* Hyperparameter tuning
* RMSE
* MAE
* Recommendation generation
* Model saving and loading
* Python scripting
* Jupyter Notebook
* Machine Learning workflow

---

# Conclusion

The **Echo Recommendation Engine** demonstrates how Collaborative Filtering can be used to build a movie recommendation system.

The project starts with raw MovieLens data, cleans and analyzes the data, creates a user-item rating matrix, and uses **SVD Matrix Factorization** to learn patterns from user ratings.

The trained model can predict ratings for movies that a user has not rated and return the **Top 5 unseen movies** based on the predicted ratings.

The final model achieved:

```text
RMSE: 0.8706
MAE:  0.6839
```

The project also includes a reusable recommendation function, sample recommendations for three users, a saved trained model, and a Python script for generating recommendations.

Overall, the project demonstrates a complete machine learning recommendation workflow from **raw data preparation to model training, evaluation, and final recommendations**.
