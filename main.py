import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# get the list of data files
csvfiles = []
for file in glob.glob("data/*.csv"):
    csvfiles.append(file)

# look at the names
csvfiles
# remove the unclean files
csvfiles.remove("data/unclean focus.csv")
csvfiles.remove("data/unclean cclass.csv")

# read  into a pandas array the files into pandas and print number of columns from each file
# also, create a new column with the name of the car
pds = []
for i, file in enumerate(csvfiles):
    pds.append(pd.read_csv(file))
    name = file[5:-4]
    pds[i]["type"] = name
    print(i, ": ", len(pds[i].columns))

# some files have 8 columns while most have 10. remove the ones that dont have 9 columns
for i, element in enumerate(pds):
    if (len(element.columns) != 10):
        pds.pop(i)

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
# linear regression
# first convert string to categories
the_data[the_data.select_dtypes(["object"]).columns] = the_data[the_data.select_dtypes(["object"]).columns].apply(lambda x: x.astype("category"))
# now create the train and test data sets
y = the_data["price"]
x = the_data.loc[:, the_data.columns != "price"]
# convert categories to dummies
x1 = pd.get_dummies(data=x, drop_first=True)
# add again the categorical type column because it will be useful later
x = pd.concat([x1, x["type"]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# fit linear regression without including the categorical column type
lm = LinearRegression()
lm.fit(x_train.loc[:, x_train.columns != "type"], y_train)
# get the r-squared value
lm.score(x_test.loc[:, x_test.columns != "type"], y_test)
# calculate predicted values for the test set
yhat = lm.predict(x_test.loc[:, x_test.columns != "type"])
# convert to a data frame with a column name
yhat = pd.DataFrame(yhat)
yhat.columns = ["yhat"]
# create data frame that contains the independent variables in the test set, the price in the test set, and the predicted values of the prices
check = pd.concat([x_test.reset_index().drop("index", 1),
                   y_test.reset_index().drop("index", 1),
                   yhat.reset_index().drop("index", 1)], axis=1)
# now plot the predicted vs actual for each car type
# the line represents x=x which is the ideal situation
# create colors to iterate through
color = iter(cm.rainbow(np.linspace(0, 1, 9)))
fig, ax = plt.subplots(3, 3)
row = 0
column = 0
for label, df in check[["yhat", "price", "type"]].groupby("type"):
    c = next(color)
    ax[row, column].scatter(x="yhat", y="price", data=df, alpha = 0.2, c=c)
    ax[row, column].axline([0, 0], [1, 1], color="black")
    ax[row, column].tick_params(axis="both", which="major", labelsize=6)
    ax[row, column].set_title(label)
    if column == 2:
        column = 0
    elif column < 2:
        column += 1
    if column == 0:
        row += 1
