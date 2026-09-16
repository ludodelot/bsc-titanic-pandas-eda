# -*- coding: utf-8 -*-
"""
Analisis exploratorio del dataset Titanic con pandas.

Actividad 7 - Programacion para Negocios (S1)
Licenciatura en Inteligencia de Negocios, Tec de Monterrey

Autor: Ludovic Delot Bravo

Originalmente desarrollado en Google Colaboratory:
https://colab.research.google.com/drive/1lFxPFD1mk2ZNFTjtLej4geXKEJ0udgEo
"""

import pandas as pd  # libreria para trabajar con archivos y dataframes
import matplotlib.pyplot as plt  # libreria para trabajar con graficas

# LECTURA DE LA BASE DE DATOS
df = pd.read_csv("data/train.csv")  # lee el archivo del Titanic
print(df.info())  # Despliega la informacion de las columnas de la bd del Titanic
print("El numero de registros y columnas de la base de datos es: ", df.shape)  # tamano de la bd
print("Los 5 primeros registros de la bd son: \n", df.head())  # despliega los 5 primeros registros de la bd
print("Los 5 ultimos registros de la bd son: \n", df.tail())  # despliega los 5 ultimos registros de la bd


# SI QUISIERA REMPLAZAR VALORES EN LAS COLUMNAS DEBO ANALIZAR PRIMERO:
print(df.isnull().sum())  # detecta los valores vacios de la bd del Titanic
print("REEMPLAZANDO LOS VALORES DE LAS COLUMNAS Age, Cabin, Embarked...")
valores = {"Age": round(df["Age"].mean(), 0), "Cabin": "Sin Registro", "Embarked": "S"}  # sacar media de columna y redondear; sustitucion por "Sin Registro" y por "S"
df.fillna(value=valores, inplace=True)  # Reemplaza por los valores solicitados
print(df.isnull().sum())  # si ya todos son cero ya no hay ninguno para reemplazar

# REEMPLAZANDO VALORES DE UNA COLUMNA ESPECIFICA
# Reemplaza los valores 1, 2 y 3 por texto en la columna de Clase
df["Pclass"].replace([1, 2, 3], ["Clase Alta", "Clase Media", "Clase Baja"], inplace=True)

df.rename(columns={"Sex": "Gender"}, inplace=True)  # Cambia el nombre de la columna
print(df.columns)  # Despliegue de columnas
print(df.info())  # Despliega la informacion de las columnas de la bd del Titanic


# Solo aplicable en datos numericos

# OBTENCION DE LOS ESTADISTICOS
print("Iniciando con la obtencion de estadisticas")
print("Los sobrevivientes del Titanic fueron: ", df["Survived"].sum())
print("Pasajeros que murieron y sobrevivieron al Titanic es: \n", df["Survived"].value_counts())
print("El numero de pasajeros por clase que subieron al Titanic es: \n", df["Pclass"].value_counts())
print("Estadisticos generales por Age y Fare son: \n ", df[["Age", "Fare"]].describe())
print("El promedio de edad de los pasajeros es de: ", df["Age"].mean())  # otras medidas: median(), mode()
print("Clasificacion por Genero y Edad: \n", df[["Gender", "Age"]].groupby("Gender").mean())
print("Costo promedio del boleto por clase es de: \n", df[["Pclass", "Fare"]].groupby("Pclass").mean())
print("Sobrevivientes por clase : \n", df[["Pclass", "Survived"]].groupby("Pclass").sum())
print("Numero de muertos y sobrevivientes por genero \n ", pd.crosstab(df.Survived, df.Gender))
print("Los titulos nobiliarios de los pasajeros del Titanic son:\n ", df.Title.value_counts())

# Usemos etiquetas que identifiquen rangos de edades
# Jovenes de 0 a 30 anos
# Adultos de 31 a 49 anos
# Adultos Mayores de 50 a 100 anos

rango = [0, 30, 49, 100]
names = ["Jovenes", "Adultos", "Adultos Mayores"]
df["Age"] = pd.cut(df["Age"], rango, labels=names)
print("Numero de Pasajeros por Edad y Clase Social \n ", pd.crosstab(df.Pclass, df.Age))

# Crea una categoria que sume el numero de miembros de la familia por pasajero
# Numero de hermanos y esposos: SibSp
# Numero de hijos y padres: Parch

df["Family_Size"] = df.SibSp + df.Parch
print("Family Size ", df)

# GRAFICOS

print("GRAFICAS")
print("Histograma")  # solamente trabaja con una columna y tiene que ser numerica
df["Survived"].plot(kind="hist", rwidth=0.5, color="lightgreen")  # rwidth es la anchura de la columna; color es el color de la columna
plt.title("Histograma de Muertos y Sobrevivientes")
plt.ylabel("Num_personas")
plt.xlabel(" Muertos (0) y Sobrevivientes (1)")
plt.legend()
plt.savefig("Grafica_Histograma_muertos_sobrevivientes.png")  # salva la grafica
plt.show()

print("Grafica de Barras Horizontales")
pd.crosstab(df.Survived, df.Gender).plot(kind="barh")
plt.title("Efecto Mortalidad y Sobrevivencia por Genero en el Titanic")
plt.xlabel("Mortalidad y Sobrevivencia del Titanic")
plt.ylabel(" Cantidad por Genero")
plt.savefig("Grafica_Barras_Muertos_sobrevivientes_Genero.png")
plt.show()

print("Grafica de Linea")
pd.crosstab(df.Pclass, df.Survived).plot(kind="line")
plt.title("Clase Social y Edad del Titanic")
plt.xlabel("Clase Social del Titanic")
plt.ylabel("Sobrevivientes")
plt.savefig("Grafica_Linea_Clase_Sobrevivientes.png")
plt.show()

print("Grafica de Pie")
colores = ["hotpink", "lightskyblue", "greenyellow", "mediumpurple"]
explode = (0.1, 0.1, 0.1)  # separacion de las graficas
plt.figure(figsize=(10, 5))  # crear lienzo
plt.pie(
    df[["Pclass", "Name"]].groupby(["Pclass"]).count(),
    explode=explode,
    labels=["Clase Alta", "Clase Media", "Clase Baja"],
    colors=colores,
    autopct="%1.0f%%",
    shadow=True,
    startangle=140,
)
plt.axis("equal")
plt.title("Pasajeros por Clase Social en el Titanic")
plt.savefig("Gráfica_Pie_Tipo_Clase.png")
plt.show()

# Grabar en un nuevo archivo todos los cambios de la bd
df.to_csv("Titanic_actualizado.csv")
