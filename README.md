# Starlight: Predicting Star Surface Temperature with Neural Networks

## Overview
This project aims to predict the surface temperature of newly discovered stars using machine learning techniques. The dataset provided by the "Nebo na Ladoni" observatory includes characteristics of 240 known stars, and we utilize this data to build a predictive model using a neural network.

The project is broken down into several key steps, including data preprocessing, exploratory data analysis (EDA), feature engineering, and training neural networks. We implemented a baseline model, followed by an optimized version using hyperparameter tuning to enhance performance.

## Project Steps

### 1. Data Preprocessing and Cleaning
- **Data Loading**: The raw dataset was loaded and checked for consistency.
- **Data Cleaning**: Removed extra spaces, standardized text cases, and eliminated redundant categories from categorical features like `star_color`.
- **Feature Engineering**: Created new features to enhance the dataset and improve model performance.

### 2. Exploratory Data Analysis (EDA)
- Analyzed quantitative and categorical features to understand their distributions and relationships.
- Created scatter plots, pair plots, and correlation heatmaps to visualize the relationships between features.

### 3. Feature Engineering
- **Logarithm Transformation**: Applied logarithm transformation to numerical features to reduce skewness and improve model performance.
- **Feature Generation**: Generated new features, such as:
  - `luminosity_relative`: Luminosity relative to the Sun.
  - `radius_relative`: Radius relative to the Sun.
  - `luminosity_radius_index` and `temperature_luminosity_index`: Interaction terms to capture complex relationships between features.

### 4. Model Training
- **Baseline Neural Network**: Built and trained a baseline model using a simple feedforward neural network with 7 hidden layers.
- **Improved Neural Network**: Built a more complex model with dropout layers, dynamic hidden layer sizes, and optimized hyperparameters.

### 5. Hyperparameter Tuning with Optuna
- Conducted hyperparameter tuning using Optuna to optimize the model architecture, including `dropout_rate`, `batch_size`, `hidden_layer_size`, and `learning_rate`.
- Best Model Parameters:
  - `dropout_rate`: 0.1066
  - `batch_size`: 8
  - `hidden_layer_size`: 512
  - `num_hidden_layers`: 4
  - `learning_rate`: 0.000235

### 6. Model Evaluation
- Evaluated the baseline and optimized models using Root Mean Squared Error (RMSE).
- **Results**:
  - Baseline Model Test RMSE: **6400**
  - Improved Model Test RMSE: **2337.38**

### 7. Conclusions
- The hyperparameter-optimized model significantly improved over the baseline, with the RMSE on the test set dropping from **6400** to **2337.38**.
- The use of `dropout` helped reduce overfitting, and the hyperparameter tuning yielded an architecture that better captured the underlying relationships in the data.

## Installation and Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/Sti11ness//starlight-temperature-prediction.git
   ```
2. Create a virtual environment (Python 3.9 recommended):
   ```sh
   python3.9 -m venv env
   ```
3. Activate the environment:
   - Windows:
     ```sh
     .\env\Scripts\activate
     ```
   - macOS/Linux:
     ```sh
     source env/bin/activate
     ```
4. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
- Open the Jupyter notebook (`starlight_temperature_prediction.ipynb`) to explore the code and results.
- To train the models with the best-found hyperparameters, run the corresponding cells in the notebook.

## Project Structure
- `starlight_temperature_prediction.ipynb`: Jupyter notebook containing the code and analysis.
- `data/`: Directory containing the dataset used in this project.
- `models/`: Directory for saving and loading trained models.
- `requirements.txt`: List of required Python packages.

## Key Files
- **`starlight_temperature_prediction.ipynb`**: Main notebook with data preprocessing, EDA, model training, and evaluation.
- **`requirements.txt`**: Specifies the required package versions for running the project.

## Results
The project aimed to predict the surface temperature of stars based on available features. The improved neural network model achieved an RMSE of **2337.38** on the test dataset, which represents a substantial improvement over the baseline.

Further improvements could involve additional feature engineering, more advanced neural network architectures, or using ensembles of models to enhance prediction accuracy.

## Future Work
- **Feature Engineering**: Generate more complex interaction terms and experiment with polynomial features.
- **Model Complexity**: Test deeper or more advanced neural network architectures, such as LSTM or transformer models, to see if they can capture additional information.
- **Transfer Learning**: Explore pre-trained models to enhance the learning process, given the limited dataset size.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements
- **"Nebo na Ladoni" Observatory** for providing the dataset.
- **Optuna** for hyperparameter tuning and optimization.

## Contact
For any questions or collaboration opportunities, feel free to reach out to [your email].

