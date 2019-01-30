import pandas 
from sklearn import linear_model
from matplotlib import pyplot
#import ha

Imdataframe = pandas.read_fwf("brain_body.txt")
List_of_x_values = Imdataframe[["Brain"]]
List_of_y_values = Imdataframe[["Body"]]

regression_object = linear_model.LinearRegression().fit(List_of_x_values, List_of_y_values)
#regression_object.fit(List_of_x_values, List_of_y_values)

#regression_object = linear_model.LinearRegression.fit(List_of_x_values, List_of_y_values)




pyplot.scatter(List_of_x_values, List_of_y_values)
pyplot.plot(List_of_x_values, regression_object.predict(List_of_x_values))
pyplot.show()