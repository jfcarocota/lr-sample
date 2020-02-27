import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
print(boston)
print("")

print("Información en el datasset:")
print(boston.keys())
print("")

print("Caracteristicas del dataset:")
print(boston.DESCR)
print("")

print("Cantidad de datos:")
print(boston.data.shape)
print("")

print("Nombre columnas:")
print(boston.feature_names)
print()

x = boston.data[:, np.newaxis, 5]
y = boston.target

#plt.scatter(x, y)
#plt.xlabel("Número de habitaciones")
#plt.ylabel("Valor medio")
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

lr = linear_model.LinearRegression()

#trainning
lr.fit(x_train, y_train)

#prediciendo
y_pred = lr.predict(x_test)

plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color = 'red', linewidth = 3)
plt.title("Regresión lineal simple")
plt.xlabel("Número de habitaciones")
plt.ylabel("Valor medio")
#plt.show()

print("")
print("Datos del modelo regresión lineal simple:")
print("")
print("Valor de pendiente o coeficiente a:")
print(lr.coef_)
print("Valor de intersección o coeficiente b:")
print(lr.intercept_)

print("")
print("Ecuación del modelo de regresión lineal:")
print('y = ', lr.coef_, 'x', lr.intercept_)

print()
print("Precisió del modelo")
print(lr.score(x_train, y_train))

plt.show()