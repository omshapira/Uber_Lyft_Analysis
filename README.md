# Uber and Lyft Analysis

The primary purpose of this project was to implement several machine learning algorithms for predicting the price of Uber and Lyft ride fares. Testing of various algorithms would help me uncover the relative impact of different predictor variables on price. The models I explored in-depth were simple/multivariate/polynomial regression, k-Nearest Neighbor Regression, Random Forest Regression, and Gradient-Boosted Regression. I also briefly tested Decision Trees and Ada-Boost Regression. Along the way, I performed various statistical tests on my models, such as the p-value test for model variable selection, and the Chow Test for evaluating coefficient variation in regression models used on weekday vs. weekend rides.

A secondary purpose of this project was to automate the creation of these models against user-defined cross-stratifications of the dataset. For example, I wanted to have the flexibility to see how these models performed on only Uber rides vs. only Lyft Rides, among only ride types that are "XL". Once these (and other) stratum parameters are defined, the notebook would recreate all exploratory analysis, visualizations, and ML model creations - each being optimally hyper-tuned. The final result would be a data frame that compares various performance metrics of each regression model. 

Click [here](https://nbviewer.jupyter.org/github/omshapira/Uber_Lyft_Analysis/blob/master/Ride_Sharing_Prices_with_code.html) to view my notebook with code blocks

Click [here](https://nbviewer.jupyter.org/github/omshapira/Uber_Lyft_Analysis/blob/master/Ride_Sharing_Prices.html) to view the same notebook without code blocks

For a slide show presentation of my analysis, check the "Machine Learning Regression Models with Uber and Lyft.pptx" file.
For a project description document, check the "Project Description.docx" file
