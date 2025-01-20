# Movie Recommendation System Based on Trends

This project is a **Movie Recommendation System** that analyzes a dataset of movies and ratings to recommend movies based on trends. It utilizes feature engineering, trend analysis, and a collaborative filtering approach to provide users with personalized movie recommendations.

---

## Table of Contents
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run the Project](#how-to-run-the-project)
- [Project Workflow](#project-workflow)
- [Code Details](#code-details)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

---

## Features
1. **Data Preprocessing**:
   - Reads movie and ratings datasets.
   - Adds a synthetic `timestamp` column to simulate time-based trends.

2. **Feature Engineering**:
   - Computes average ratings and the number of ratings for each movie.
   - Calculates weekly trends and growth rates for ratings.

3. **Model Training**:
   - Uses a collaborative filtering approach (e.g., Singular Value Decomposition) to train a recommendation model.

4. **Trend-Based Recommendations**:
   - Combines rating statistics and growth trends to recommend trending movies.

---

## Folder Structure
```
project/
|-- data/
|   |-- movies.csv        # Movie dataset
|   |-- ratings.csv       # Ratings dataset
|
|-- src/
|   |-- data_preprocessing.py # Preprocess and load data
|   |-- feature_engineering.py # Generate features from data
|   |-- model_training.py      # Train the recommendation model
|   |-- trend_recommendation.py # Generate recommendations
|
|-- main.py              # Entry point for the project
|-- requirements.txt     # Python dependencies
|-- README.md            # Project documentation
```

---

## Prerequisites
- Python 3.10 or higher
- Recommended IDE: Visual Studio Code or PyCharm

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/movie-recommendation-trends.git
   cd movie-recommendation-trends
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your datasets (`movies.csv` and `ratings.csv`) inside the `data/` folder.

---

## How to Run the Project
1. Run the main script:
   ```bash
   python main.py
   ```

2. View the recommendations generated in the terminal or as output files.

---

## Project Workflow
1. **Data Loading**: Load the movies and ratings datasets.
2. **Data Preprocessing**: Add a dummy `timestamp` column for trend analysis.
3. **Feature Engineering**:
   - Compute rating statistics.
   - Calculate weekly trends and growth rates.
4. **Model Training**:
   - Train a collaborative filtering model (SVD from `scikit-learn`).
5. **Generate Recommendations**:
   - Combine trend analysis and collaborative filtering to recommend movies.

---

## Code Details
### main.py
Coordinates the workflow:
- Loads and preprocesses data.
- Extracts features.
- Trains the model.
- Generates recommendations.

### src/data_preprocessing.py
Handles:
- Loading datasets.
- Adding a synthetic timestamp column.

### src/feature_engineering.py
Generates:
- Average ratings, number of ratings.
- Weekly trends and growth rates.

### src/model_training.py
Trains a collaborative filtering model using the ratings data.

### src/trend_recommendation.py
Combines:
- Collaborative filtering results.
- Trend analysis to generate the final recommendations.

---

## Output
The output will consist of recommended movies ranked by:
- High ratings.
- Increasing popularity trends.

Example output:
```
movie_id,movie_title,avg_rating
63,A Clockwork Orange,3.36
67,The Silence of the Lambs,3.34
12,The Lion King,3.30
```

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

