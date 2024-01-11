import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

nba = pd.read_csv('nba_games.csv')
#print(nba.info())

# Analyzing relationship between quantative and categorical values

nba_10 = nba[nba.year_id == 2010]
nba_14 = nba[nba.year_id == 2014]

nets_pts_10 = nba_10[nba_10.fran_id == 'Nets'].pts
knicks_pts_10 = nba_10[nba_10.fran_id == 'Knicks'].pts

diff_means_2010 = round(abs(np.mean(nets_pts_10) - np.mean(knicks_pts_10)),2)
#print(diff_means_2010)

plt.hist(nets_pts_10, color = 'blue', density = True, label = 'Nets', alpha = 0.5)
plt.hist(knicks_pts_10, color = 'orange', density = True, label = 'Knicks', alpha = 0.5)
plt.show()
plt.clf()

nets_pts_14 = nba_14[nba_14.fran_id == 'Nets'].pts
knicks_pts_14 = nba_14[nba_14.fran_id == 'Knicks'].pts

diff_means_14 = round(abs(knicks_pts_14.mean() - nets_pts_14.mean()),2)

plt.hist(nets_pts_14, color = 'blue', density = True, alpha = 0.5)
plt.hist(knicks_pts_14, color = 'orange', density = True, alpha = 0.5)
plt.show()
plt.clf()

sns.boxplot(data = nba_10 ,y = 'pts', x = 'fran_id')
plt.show()

# By comparing the different values between the franchise id and the amount of points scored there seems to be enough information to say there is a association between the franchise and the points scored.

# Analyzing relationship between categorical variables
location_result_freq = pd.crosstab(nba.game_result, nba.game_location)
#print(location_result_freq)

#initial resusts of the contingency table suggest that there is a pretty strong relationship between game location and game result.

location_result_prop  = location_result_freq / len(nba.game_result)

# the proportions of the contingency table provide a clearer picture and reinforce the initial observation

#print(location_result_prop)

chi2, pval, dof, expected = chi2_contingency(location_result_freq) 
# print(expected)
# print(chi2)

# the Chi squared statistic returns 1359.29, even further suggesting a strong relationship between location and result.

# Analyzing Relationship between quatitative values

point_diff_forecast_cov = np.cov(nba_10.point_diff, nba_10.forecast)
#print(point_diff_forecast_cov[0][1])
# the covariance is very close to one suggesting that there is not much of a relationship between the forecast and the point difference. although not negative, not a high positive by any means.

point_diff_forecast_corr = pearsonr(nba_10.point_diff, nba_10.forecast)
print(point_diff_forecast_corr)

#the correlation coefficient of .44 suggest that there is a relationship between the  point difference and the forecast but not a strong one. 

plt.scatter(x = nba_10.forecast, y = nba_10.point_diff)
plt.ylabel('Point Difference')
plt.xlabel('Forecast')
plt.show()
plt.clf()
# the graph seems to follow the result of both the covariance and correlation coefficient in that there is a positive relationship but not a strong one.