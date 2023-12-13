![](![UTA-DataScience-Logo](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/329ebeed-5741-4bbe-862e-ebcfdc50e4c3)
)
![UTA-DataScience-Logo](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/a0f724b6-ce3f-40d7-aa51-431f3c04af69)

# Bike Sharing Demand

* **Summary** : This repository holds an attempt to combine historical usage patterns with weather data
* in order to forecast bike rental demand in the Capital Bikeshare program in Washington, D.C (https://www.kaggle.com/c/bike-sharing-demand). 

## Overview

  * **Definition of the tasks / challenge**  :The task, as defined by the Kaggle challenge is to develop a predictive model that combines historical bike-sharing usage patterns with weather data to forecast the demand for bike rentals in the Capital Bikeshare program in Washington, D.C.
  * **Your approach** : The analysis is divided into two parts: the initial exploration focuses on discovering patterns between variables, while the second part delves into model building, progressing from basic regression to methods such as ridge/lasso regression, random forest and gradient boosting methods.

  * **Summary of the performance achieved** : The random forest model emerged as the most effective among the models explored, with an RMSLE value of 0.1030. For my submission, the RMSLE score obtained was 0.42499, and the top-performing model on Kaggle had a RMSLE score of 0.33756.


## Summary of Workdone


### Data

* Data:
    * Input: train.csv file
         * datetime - hourly date + timestamp  
         * season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
         * holiday - whether the day is considered a holiday
         * workingday - whether the day is neither a weekend nor holiday
         * weather -
               * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
               * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
               * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
               * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
         * temp - temperature in Celsius
         * atemp - "feels like" temperature in Celsius
         * humidity - relative humidity
         * windspeed - wind speed
         * casual - number of non-registered user rentals initiated
         * registered - number of registered user rentals initiated
         * count - number of total rentals
  * Size: 1.12 MB
  * Instances: We are provided hourly rental data spanning two years. The training set is comprised of the first 19 days of each month, while the test set is from the 20th to the end of the month.
       * - 6493 instances in test.csv
       * - 10886 instances in train.csv
       * - 6493 instances in submission.csv 


#### Preprocessing / Clean up

1) Feature Engineering:
   Created new features from the "datetime" column, such as "date," "hour," "weekday," "month," and mapped "season" and "weather" values to descriptive categories.
   Converted categorical variables ("hour," "weekday," "month," "season," "weather," "holiday," "workingday") to the category data type.

2) Data Visualization:
   Plotted a matrix of missing values using msno.matrix to visualize the completeness of the data.
   Created subplots for box plots to visualize the distribution of the target variable "count" across different features.
   Box plots were created for "count" alone, "count" across seasons, "count" across hours of the day, and "count" across working days.

3) Outlier Removal:
   Detected and removed outliers using the criteria that the absolute difference between "count" and its mean should be within three times the standard deviation.
   Displayed the shapes of the data before and after outlier removal.
   The original dataset had a shape of (10886, 15), and after removing outliers, the shape became (10739, 15).



#### Data Visualization

Correlation Matrix:

   1)   Temperature and Humidity Correlation:
          Temperature ("temp") has a positive correlation with the target variable "count," while humidity has a negative correlation.
          

   2)   Windspeed:
           The correlation suggests that "count" and windspeed are not strongly related.
           
   3)   atemp and temp Multicollinearity:
            "atemp" is not considered since it has a strong correlation with "temp."
            "atemp" or "temp" should be dropped.
            
            
![corrematrix](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/b4732768-bed8-464b-ab93-13cefb2629ca)

Box Plots:

   1)  Spring Season and Lower Count: Spring season has a relatively lower count and the dip in the median value in the box plot provides evidence for this observation.
   
   2)  Hour of the Day Boxplot: The boxplot analysis of the "Hour of the Day" has higher median values observed at 7 AM - 8 AM and 5 PM - 6 PM. This can be attributed to these peaks in regular school and office users during these times.
   
   3)  Outliers and Working Days: Most of the outlier points are contributed by "Working Day" rather than "Non-Working Day" and can be seen evidently in Figure 4.
   
   
![boxplots](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/a7ae86e0-bad4-4bf4-8510-bf8fb431f94f)


More Visualizations:

![Screenshot 2023-12-12 220313](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/eef44982-c821-4a61-aa18-9d4195b6d44c)



![Screenshot 2023-12-12 220332](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/8471f3ec-2a3e-4c5a-8c2a-a4e5b0b9d9dd)



![Screenshot 2023-12-12 220359](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/1c418090-fd18-4bd7-bb64-b24bbd2eeec5)



![Screenshot 2023-12-12 220426](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/608411c4-6a71-4b96-bf64-e120d99749cb)



### Problem Formulation

* Define:
  * Input: Hourly rental data spanning two years, including features such as datetime, season, holiday, workingday, weather, temperature, humidity, windspeed, etc.
  * Output: Prediction of the total count of bikes rented during each hour covered by the test set.
 
* Models:


  * 1) Linear Regression:
  A simple model assuming a linear relationship between input features and bike rental counts.
  Used as a baseline model due to its simplicity and interpretability.
       * Loss: Mean Squared Error (MSE).
       * Optimizer: Ordinary Least Squares (OLS).

  * 2) Ridge Regression and Lasso Regression:
   Regularized linear regression models are used to prevent overfitting by penalizing large coefficients.
   Employed to handle potential multicollinearity and prevent overfitting.
        * Loss: Mean Squared Error (MSE) with regularization term.
        * Optimizer: Gradient Descent.
          

   * 3) Random Forest:
    Ensemble model combining multiple decision trees to capture complex relationships in the data.
    Chosen for its ability to handle non-linear relationships and capture interactions between features.
        * Loss: Mean Squared Logarithmic Error (RMSLE).
        * Optimizer: Not applicable (ensemble method).


   * 4) Gradient Boosting:
     Ensemble model building trees sequentially, with each tree correcting the errors of the previous one.
     Applied for its robustness and ability to improve predictive accuracy by sequentially minimizing errors
         * Loss: Mean Squared Logarithmic Error (RMSLE).
         * Optimizer: Gradient Boosting (ensemble method).



### Training

* Describe the training:
  * Overfitting: Used techniques like regularization to address overfitting.
  * Hyperparameter Tuning: Grid search was used.
  * Data Quality: Addressed issues like missing or inconsistent data through preprocessing.
    

##  Key Performance Metric

* Key Performance Metric: Root Mean Squared Logarithmic Error (RMSLE) 
* ![Screenshot 2023-12-12 232919](https://github.com/itskhushij/DATA-3402---Fall-2023/assets/86798086/86604b75-1557-497e-80a5-142902e88926)

### Conclusions

* In the context of bike-sharing demand prediction, the random forest was identified as the best model, with a RMSLE value of 0.103. The model's ability to capture complex patterns in the data implies that it can effectively identify peak usage times when the demand for bikes is the highest. This information is valuable for bike-sharing service providers to ensure they have a sufficient number of bikes available during high-demand periods.


### Future Work

* Feature Engineering

   * Explore additional features or combinations of features like incorporating external data sources, such as weather forecasts or events happening in the area.

  
* Time series analysis:

   * Identify patterns and trends within the data to understand how the bike rental count changes over time, whether there are regular ups and downs, and if there are any seasonal variations.



### Overview of files in repository

*  Directory structure
  * The BikeSharingKaggle.ipynb notebook contains the complete code for data preprocessing, exploratory data analysis, feature engineering, model training (including linear regression, ridge regression, lasso regression, random forest, and gradient boosting), 
     evaluation metrics, and visualizations for the bike sharing demand prediction task.
    
    

### Data

* Download the data from the Kaggle Challenge (https://www.kaggle.com/competitions/bike-sharing-demand/overview)


### Overview

* 1) Loaded the bike sharing dataset, including features like datetime, season, weather, temperature, and more.
* 2) Preprocessed the datetime column to extract additional features like hour, weekday, and month.
* 3) Plotted visualizations to explore the distribution of bike sharing counts across different factors, such as season, hour, and working day.
* 4) Created box plots to identify outliers and understand the data distribution.
* 5) Removed outliers from the dataset to improve the model.
* 6) Conducted correlation analysis to understand the relationships between variables like temperature, humidity, and bike counts.
* 7) Converted categorical variables (e.g., season, weather) into the appropriate data type for training.
* 8) Split the dataset into training and testing sets.
* 9) Defined the input features and the target variable (bike counts).
* 10) Used the Random Forest Regressor from the scikit-learn library.
* 11) Trained the model on the training data using the fit method.



#### Performance Evaluation

* 1) Evaluated the model's performance using the Root Mean Squared Logarithmic Error (RMSLE) metric.
* 2) Computed the RMSLE value on the testing set to assess how well the model generalized to unseen data.
* 3) Concluded that the Random Forest model outperformed other models based on the achieved RMSLE score.



## Citations

* (https://www.kaggle.com/code/viveksrinivasan/eda-ensemble-model-top-10-percentile/notebook#Feature-Engineering)https://www.kaggle.com/code/viveksrinivasan/eda-ensemble-model-top-10-percentile/notebook#Feature-Engineering)







