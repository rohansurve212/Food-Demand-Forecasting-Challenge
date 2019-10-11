Food_Demand_Forecasting_Challenge
==============================

Problem Statement
Your client is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.

The replenishment of majority of raw materials is done on weekly basis and since the raw material is perishable, the procurement planning is of utmost importance. Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful. Given the following information, the task is to predict the demand for the next 10 weeks (Weeks: 146-155) for the center-meal combinations in the test set:

Historical data of demand for a product-center combination (Weeks: 1 to 145)
Product(Meal) features such as category, sub-category, current price and discount
Information for fulfillment center like center area, city information etc.
Data Dictionary
Weekly Demand data (train.csv): Contains the historical demand data for all centers, test.csv contains all the following features except the target variable.
Variable	Definition
id	Unique ID
week	Week No
center_id	Unique ID for fulfillment center
meal_id	Unique ID for Meal
checkout_price	Final price including discount, taxes & delivery charges
base_price	Base price of the meal
emailer_for_promotion	Emailer sent for promotion of meal
homepage_featured Meal	featured at homepage
num_orders	(Target) Orders Count
fulfilment_center_info.csv: Contains information for each fulfilment center
Variable	Definition
center_id	Unique ID for fulfillment center
city_code	Unique code for city
region_code	Unique code for region
center_type	Anonymized center type
op_area	Area of operation (in km^2)
meal_info.csv: Contains information for each meal being served
Variable	Definition
meal_id	Unique ID for the meal
category	Type of meal (beverages/snacks/soups….)
cuisine	Meal cuisine (Indian/Italian/…)
Evaluation Metric
The evaluation metric for this competition is 100*RMSLE where RMSLE is Root of Mean Squared Logarithmic Error across all entries in the test set.

Test data is further randomly divided into Public (30%) and Private (70%) data.

Solution
    Converted this time series problem to regression problem.

    
Data transformation:
Here number of orders placed (target variable) is highly right skewd so that Log transformation is applied.
Log transformation of base_price, checkout_price, and num_orders.

Feature engineering:
For every record difference between base_price and checkout_price.
Differenc of previous week checkout_price and current weeks checkout_price.
Lag features of 10,11, and 12 week lagging features. Here I have used lag of last 10 weeks because we have to predict for 10 weeks in test dataset.
Exponentially weighted mean over last 10, 11, and 12 weeks.
Cross validation strategy:
Last 10 weeks (136 - 145) of every center-meal pair data is used as a Validation dataset from train dataset.
Model
One single CatBoost model which has RMSLE of 0.54.
High regularization so it does not overfit because of new features made using target variable.
What didn't work:
just using original data as it is and using catboost regressor gave RMSLE of 1.58
Only using difference between base_price and checkout_price, difference between base_price and checkout_price as a features and not using any lag and exponentially weighed features didn't give good score.
Rolling mean and median over last 26, 52, 104 weeks as features didn't work that well, feature importance was low.
Geographical features had low feature importance, So didn't use them in final model.
TODO / Improvements:
Extensive hyper parameter tuning and feature selection.
Create more features based on Categorical Encoding methods like mean encoding, freqvency encoding, hash encoding etc.
Try more algorithms like xgboost, LightGBM, Linear Regression etc.
Try ARIMA , Prophet etc.
Ensemble of different models.