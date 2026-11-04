# Databootcamp-Regression-Competition-for-house-prices
Team members: Melody Zuo and Yihuan Zhang

## Project Overview

We worked with a dataset through handling missing values and testing several models including Linear Regression, Polynomial Regression, Ridge Regression, and Random Forest. We also created additional features based on house size, age, overall quality, and garage spaces. After comparing different models using cross-validation and Kaggle public scores, we selected a Ridge Regression model with `alpha=10`.

Final Model Kaggle Public RMSE: 27,003.58788

## Process Description

We started with all 79 features from the original features, leaving out Id because it only identifies each house and SalePrice because it is what we want to predict. Then, We used pandas to calculate six additional features from the provided columns.

1. TotalArea = TotalBsmtSF + 1stFlrSF + 2ndFlrSF.Reason: one number summarizes the basement and main floors, so the model can use an overall size signal rather than learning three separate effects only.
2. TotalBathrooms = FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath.Reason: a half bath offers less than a full bath; this gives one approximate measure of bathroom capacity while retaining the original columns too.
3. HouseAge = YrSold − YearBuilt.Reason: age when sold is more directly interpretable than construction year alone and may capture wear or differences between old and new homes.
4. QualityLivingArea = OverallQual × capped GrLivArea. Reason: extra living space may be worth more in a higher-quality home. GrLivArea is first capped at 4,000 square feet to reduce extreme linear extrapolation. This is a small interaction of the kind introduced by PolynomialFeatures.
5. LargeEdwards = 1 if Neighborhood is Edwards and original GrLivArea exceeds 4,000; otherwise 0. Reason: price patterns for unusually large homes may vary by neighborhood, so we tested a separate adjustment for this observable combination. Training support is limited (one matching row), so the estimated effect may overfit.
6. QualityGarageCars = OverallQual × GarageCars. Reason: a garage space may have a different price contribution in homes of different overall quality. We tested this single interaction after broader polynomial terms and other small interactions. Five-partition mean RMSE fell from 27,347.14 to 27,184.01, and Kaggle public RMSE fell from 27,643.51679 to 27,003.58788.
   
We treated then MSSubClass as a categorical variable, capped GrLivArea at 4,000, filled missing values, standardized numerical features, and one-hot encoded categorical features. The model therefore starts with 85 input columns (79 original + 6 created), but one-hot encoding produces more columns internally.

## Models improving process

1. Starting with the class baseline - we first made sure we were using the updated data correctly, predicting the right target, and keeping the test rows in the correct order. Our first LinearRegression model used only OverallQual and had a 20% holdout RMSE of 49,041.38. We then tried 20 original predictors with missing-value filling and categorical encoding, improving RMSE to 45,628.45. An exploratory all-feature model also achieved an out-of-fold RMSE of 35,388.55 on one five-fold partition.Using more predictors improved our results. Since ordinary linear regression can be unstable when predictors are strongly related, we decided to try Ridge next.

2. Trying polynomial features - House prices may not increase in a perfectly linear way with features like size and quality. Since we learned about squared terms and interactions using PolynomialFeatures, we tested whether adding them would improve predictions. We added degree-two terms for five numeric predictors: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, and YearBuilt. Ridge helped control the larger feature set. Across three five-fold partitions, alpha=30 achieved an average RMSE of 31,791.90, compared with 31,856.24 for alpha=100 and 31,996.16 for alpha=300. However, adding many quadratic terms did not perform as well as the simpler Ridge models we tested later. We decided to focus on a few more meaningful interactions instead. We did not obtain a valid Kaggle score for this polynomial model.

3. Comparing with a random forest- We also tested RandomForestRegressor to compare a tree-based model with linear regression, especially for unusually large houses. We tested different numbers of features per split, minimum leaf sizes, tree depths, and feature subsets. The all-feature model with max_features=0.3 and min_samples_leaf=1 achieved a mean RMSE of 29,478.94 across two five-fold partitions. A 20-feature version achieved 29,702.21 across three partitions. Additional tests with 400 trees also showed that max_features=0.3 performed best among the three settings we compared. We then trained 800 trees on all labeled rows. Its Kaggle public RMSE was 30,697.63354. Although useful as a comparison, it was not our best model.

4. Switching to Ridge and adding simple features We had previously learned Ridge regression in class. It is similar to linear regression, but limits overly large coefficients, which helps when predictors overlap.
We used the original numeric and categorical predictors, excluding Id and SalePrice. We filled missing numerical values with medians, labeled missing categories as "Missing," converted categories using one-hot encoding, and standardized numerical data. We also created three additional features: total area, total bathrooms, and house age. Ridge with alpha=10 achieved an average RMSE of 29,300.47 across three five-fold partitions. Adding an interaction between OverallQual and GrLivArea improved it to 29,202.54. We kept Ridge and the quality-by-living-area interaction for further testing. These versions did not have valid Kaggle scores.

5. Capping unusually large living areas We noticed that very large houses could cause a linear model to predict unusually high prices. We tested different limits on GrLivArea without removing any training rows. In a later Ridge setup, leaving living area uncapped produced an average RMSE of 29,206.6. Capping it at 3,500 square feet improved RMSE to 28,670.2, while a 4,000-square-foot cap achieved 28,327.59. However, the 4,000-square-foot model scored 54,417.66090 on Kaggle, much worse than its cross-validation result. We could see an unusually large test home and a high prediction, but could not confirm its actual sale price or identify it as the cause of the error. This showed us that good cross-validation results do not always translate into good Kaggle performance.

6. Adding the LargeEdwards indicator - During our analysis, we noticed that very large homes could have different prices depending on their neighborhood. We tested an additional feature to capture this. LargeEdwards equals 1 when a home is in Edwards and its original GrLivArea exceeds 4,000 square feet. Otherwise, it equals 0. This allows the model to account for this specific combination separately. Keeping the 4,000-square-foot cap and Ridge(alpha=10), our average RMSE across three five-fold partitions improved from 28,327.59 to 27,398.92. The Kaggle public RMSE also improved to 27,643.51679. We kept this as our best confirmed result at that point. Limitation: Only one training row meets the LargeEdwards condition, so its effect is uncertain and could lead to overfitting.

7. Averaging Ridge and random forest predictions - Although Ridge performed better on Kaggle, we thought the random forest might help correct some of its prediction errors, especially for large houses. We tested combining their predictions, giving Ridge more weight.
A 75% Ridge and 25% random forest combination achieved an average RMSE of 26,728.83 across three five-fold partitions, compared with 27,398.92 for Ridge alone and 29,632.94 for the forest alone. However, its Kaggle score was 27,802.62697, worse than Ridge's 27,643.51679. We rejected the combination because it performed better locally but worse on the actual leaderboard. Testing other feature changes Using our best Ridge model as the reference, we tested a few additional changes.

8. Ordered quality encoding - We replaced one-hot encoding with ordinal encoding for three, five, or eight quality-related columns. Their average RMSEs were 28,079.0, 27,988.2, and 27,951.8, respectively, all worse than the one-hot reference of 27,398.9. We kept one-hot encoding.

Additional interactions: We tested different combinations of quality and garage size. A version with two garage-related interactions performed slightly better locally, but the improvement was small. We chose the simpler single-feature QualityGarageCars interaction instead and tested it on two additional partitions.


## Choosing our final Ridge model 

Our final additional feature was QualityGarageCars, calculated as OverallQual × GarageCars. This captures how the effect of garage size may differ depending on a home's overall quality. It uses the polynomial-feature idea we learned in class without adding many unnecessary terms.
Across five five-fold partitions, the final model achieved an average RMSE of 27,184.01, compared with 27,347.14 for the previous Ridge model on the same partitions. It improved all five partitions, although the average improvement was relatively small.

We then tested it on Kaggle. Its public RMSE was 27,003.58788, improving on our previous score of 27,643.51679 by 639.92891.
