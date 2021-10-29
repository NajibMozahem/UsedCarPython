import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from matplotlib.pyplot import cm
import seaborn as sns
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# get the list of data files
csvfiles = []
for file in glob.glob("data/*.csv"):
    csvfiles.append(file)

# look at the names
csvfiles
# remove the unclean files
csvfiles.remove("data\\unclean focus.csv")
csvfiles.remove("data\\unclean cclass.csv")

# read  into a pandas array the files into pandas and print number of columns from each file
# also, create a new column with the name of the car
pds = []
for i, file in enumerate(csvfiles):
    pds.append(pd.read_csv(file))
    name = file[5:-4]
    pds[i]["type"] = name
    print(i, ": ", len(pds[i].columns))

# some files have 8 columns while most have 10. remove the ones that dont have 10 columns

pds = [x for x in pds if not len(x.columns) != 10]

# check the number of columns in each panda
for element in pds:
    print(len(element.columns))
# everything looks fine

# let us now look at the names of the columns
for element in pds:
    print(element.columns)
# we note that in one case the tax column is labelled differently. We need the same labels.
# get the column names from first data frame
names = pds[0].columns.tolist()
# now loop thrpugh the data frames setting the names to this variable:
for element in pds:
    element.columns = names

# now append the data frames into a single data frame
the_data = pd.DataFrame()
for element in pds:
    the_data = the_data.append(element, ignore_index=True)

# check for missing values
the_data.isnull().sum()

# summarise the data
the_data.describe()
# it seems that the maximum year is 2060. This doesnt make sense.
the_data[the_data["year"] > 2020]
# we should remove this record
the_data = the_data[the_data["year"] < 2020]

# Visualise the data

sns.displot(the_data["price"])
# we see that the price is highly skewed. Produce the same but on a log scale:
sns.displot(the_data["price"], log_scale=True)

# Let us now look at the frequency of the car types in the data set
the_data["type"].value_counts().plot(kind="bar")

# Let us now look at the percentage of each car in terms of year. This will help us
# understand whether the data set contains some car types that are newer than other car types

# first we need to put the data in the appropriate shape
# count the number of cars for each type in each year

type_year_count = the_data[["year", "type"]].groupby(["type", "year"]).size().reset_index(name="counts")
# pivot the table to wide format
type_year_count = type_year_count.pivot_table(index="type", columns="year", values="counts")
# replace nan values by 0
type_year_count = type_year_count.replace(np.nan, 0)
# make column names a string instead of a number
type_year_count.columns = list(map(str, type_year_count.columns))
# get the names of the columns
years = type_year_count.columns
# convert the columns to int instead of float
type_year_count[years] = type_year_count[years].astype(int)
# calculate percentages
type_year_count["total"] = type_year_count.sum(axis=1)
for column in years:
    type_year_count[column] = type_year_count[column]/type_year_count["total"]
# drop the total column
type_year_count.drop("total", 1, inplace=True)
# now plot
type_year_count.plot(kind="bar", stacked = True)
plt.legend(ncol=5, loc = "upper left")

# we now produce the same graph but for fuel type
type_fuel_count = the_data[["fuelType", "type"]].groupby(["type", "fuelType"]).size().reset_index(name="counts")
# pivot the table to wide format
type_fuel_count = type_fuel_count.pivot_table(index="type", columns="fuelType", values="counts")
# replace nan values by 0
type_fuel_count = type_fuel_count.replace(np.nan, 0)
# get the names of the columns
fuel = type_fuel_count.columns
type_fuel_count = type_fuel_count.astype(int)
# calculate percentages
type_fuel_count["total"] = type_fuel_count.sum(axis=1)
for column in fuel:
    type_fuel_count[column] = type_fuel_count[column]/type_fuel_count["total"]
# drop the total column
type_fuel_count.drop("total", 1, inplace=True)
# now plot
type_fuel_count.plot(kind="bar", stacked = True)
plt.legend(ncol=5, loc = "upper left")

# we now look at differences in price when taking into account the car type and the year
# first we need to get the data in the correct format
# calculate the average price for each type and year combination
heat = the_data[["price", "type", "year"]].groupby(["type", "year"]).mean()
# now pivot the table to wide format
heat = heat.pivot_table(index = "type", columns="year", values="price")
# now plot
sns.heatmap(heat.transpose(), cmap = sns.cm.rocket_r)
# the plot shows that the most expensive cars are audi. bmw, and
# mercedes. However, we also see a significant drop in their prices
# as they get older. We do not see the same steep decrease in
# price for ford and hyundai.

# we now look at the change in price as a function of mileage:
sns.lmplot(x="mileage", y="price", hue="type", scatter=False, lowess=True, ci=None, data=the_data)

# let us look at the correlation between price and the other variables
the_data.corr().sort_values("price").loc[:, "price"][the_data.corr()["price"] != 1.0].plot(kind="bar")
plt.xticks(rotation=0)

# first convert string to categories
the_data[the_data.select_dtypes(["object"]).columns] = the_data[the_data.select_dtypes(["object"]).columns].apply(lambda x: x.astype("category"))

# now transform the variables in the train sets using a mapper
# i transform before splitting in order to have same columns in train and test sets
mapper = DataFrameMapper([
    (["year"], sklearn.preprocessing.StandardScaler()),
    (["mileage"], sklearn.preprocessing.StandardScaler()),
    (["tax"], sklearn.preprocessing.StandardScaler()),
    (["mpg"], sklearn.preprocessing.StandardScaler()),
    (["engineSize"], sklearn.preprocessing.StandardScaler()),
    ("type", sklearn.preprocessing.LabelBinarizer()),
    ("fuelType", sklearn.preprocessing.LabelBinarizer()),
    ("transmission", sklearn.preprocessing.LabelBinarizer()),
    ("model", sklearn.preprocessing.LabelBinarizer()),
    ("price", None)
], df_out=True)
the_data_transformed = mapper.fit_transform(the_data)
# add again the categorical type column because it will be useful later and we need it for
# the stratified sampling since this column represents the groups
the_data_transformed = pd.concat([the_data_transformed, the_data["type"]], axis=1)
# now split the data set using stratified sampling using the variable type
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
for train_index, test_index in split.split(the_data_transformed, the_data_transformed["type"]):
    strat_train_set = the_data_transformed.iloc[train_index]
    strat_test_set = the_data_transformed.iloc[test_index]

# now create copies of train and test sets for x and y
x_train = strat_train_set.drop(["price", "type"], axis=1)
y_train = strat_train_set["price"]
x_test = strat_test_set.drop(["price", "type"], axis=1)
y_test = strat_test_set["price"]

# linear regression
lm = LinearRegression()
lm.fit(x_train, y_train)
# calculate predicted values for the test set
yhat_lm = lm.predict(x_test)
# get the RMSE
np.sqrt(mean_squared_error(y_test, yhat_lm))

# Decision tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
yhat_tree = tree_reg.predict(x_test)
np.sqrt(mean_squared_error(y_test, yhat_tree))
tree_reg.score(x_test, y_test)

# the RMSE for the decision tree is much smaller. We therefore go with it. It also has a good R2 value
# on the test set

# convert yhat to data frame to concatenate it with other data frames for producing the figures later
yhat_tree = pd.DataFrame(yhat_tree)
yhat_tree.columns = ["yhat_tree"]
# create data frame that contains the independent variables in the test set, the price in the test set, and the predicted values of the prices
check = pd.concat([x_test.reset_index().drop("index", 1),
                   strat_test_set["type"].reset_index().drop("index", 1),
                   y_test.reset_index().drop("index", 1),
                   yhat_tree.reset_index().drop("index", 1)], axis=1)
# now plot the predicted vs actual for each car type
# the line represents x=x which is the ideal situation
# create colors to iterate through
color = iter(cm.rainbow(np.linspace(0, 1, 9)))
fig, ax = plt.subplots(3, 3)
row = 0
column = 0
for label, df in check[["yhat_tree", "price", "type"]].groupby("type"):
    c = next(color)
    ax[row, column].scatter(x="yhat_tree", y="price", data=df, alpha = 0.2, c=c)
    ax[row, column].axline([0, 0], [1, 1], color="black")
    ax[row, column].tick_params(axis="both", which="major", labelsize=6)
    ax[row, column].set_title(label)
    if column == 2:
        column = 0
    elif column < 2:
        column += 1
    if column == 0:
        row += 1





