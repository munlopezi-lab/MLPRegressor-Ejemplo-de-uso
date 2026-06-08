import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargamos el dataset
dataset = pd.read_csv('insurance.csv')

# Convertimos las variables categóricas a numéricas
le = LabelEncoder()
dataset['smoker'] = le.fit_transform(dataset['smoker'])         # smoker: si=1, no=0
dataset['sex'] = le.fit_transform(dataset['sex'])               # sex: male=1, female=0
'''la siguiente línea se explica mejor en el reporte, pero básicamente se encarga de crear nuevas columnas para 
cada región (norte, sur, este, oeste) y asignar un valor de 1 o 0 dependiendo de si la fila corresponde o nó a 
la región...'''
dataset = pd.get_dummies(dataset, columns=['region'], drop_first=True,dtype=int)   # asignamos variables numericas a region y sex

print(dataset.filter(like='region'))    #<-- esto es para verificar que se hayan creado correctamente los vectores corespondientes a cada región

 # verificamos que se hayan convertido correctamente las variables categóricas 

X=dataset.iloc[:,0:6].values    #extraemos las primeras 6 columnas como características
Y=dataset.iloc[:,-1].values     #extraemos la última columna como variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)   #dividimos el dataset en entrenamiento y prueba

scaler = StandardScaler()   #estandarizamos las medidas para mejorar el rendimiento de la red
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#para comparar usaremos un bucle que itere sobre diferentes funciones de activación y 
# guarde los resultados en un diccionario para luego presentarlos en una tabla

activaciones = ['relu', 'logistic', 'tanh', 'identity']     #funciones de activación a comparar

resultados = []     #aquí guardaremos los resultados de cada función de activación

print("Comparación de funciones de activación:\n")

plt.figure(figsize=(10, 50))     # preparamos una figura para graficar los resultados
for f in activaciones:
    # Definir el modelo con la función actual
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 6),  # Dos capas ocultas con 100 y 6 neuronas
        activation=f,
        random_state=42,
        max_iter=10000,
        learning_rate_init=0.01,
        solver='adam'
    )
    
    # Entrenamos el modelo
    mlp.fit(X_train_scaled, y_train)
    
    # Predecimos con el conjunto de prueba
    y_pred = mlp.predict(X_test_scaled)
    
    # Error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)
    # Error absoluto medio
    mae = mean_absolute_error(y_test, y_pred)
    
    
    # Guardar los  resultados
    resultados.append({
        'Activación': f,
        'MSE': mse,
        'MAE': mae,
        
    })

    plt.plot(mlp.loss_curve_, label=f'Activación: {f}') # Graficamos la curva de pérdida para cada función de activación

# mostramos resultados en una tabla
df_resultados = pd.DataFrame(resultados)
print("-" * 50)
print(df_resultados.to_string(index=False))
print("-" * 50)

# Graficamos la comparación de las curvas de pérdida
plt.title('Loss vs Épocas por Función de Activación')
plt.xlabel('Épocas ')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig('curva_perdida_comparativa.png')
plt.show()
