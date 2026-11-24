<p align="center"><b><font size="5">Predicción de Churn de telecomunicaciones con modelos clásicos de machine learning y redes neuronales</font></b></p>

**Resultados principales:**

-	Recall de 0.97 en producción
-	XGBoost como mejor modelo
-	Aplicación desplegada en Streamlit

**Descripción y meta del proyecto:**

La meta del proyecto es predecir el abandono de clientes de una compañía de Telecomunicaciones y desarrollar una aplicación usando modelos de machine learning y redes neuronales. La métrica principal para evaluar los modelos fue el recall. 
El resultado es una aplicación que devuelve la probabilidad de un cliente de dejar o quedarse con el servicio. A esta aplicación se le podrá introducir los parámetros para la inferencia o generarlas al azar.

<img width="295" height="187" alt="image" src="https://github.com/user-attachments/assets/868cabe2-b381-4347-abda-a7291132c576" />
<img width="288" height="188" alt="image" src="https://github.com/user-attachments/assets/2688804a-5ab3-4664-a87f-36688ce31b0f" />

**Dataset:** 

El dataset usado es de libre acceso de Machine learning repository: Dataset de 12 meses de una compañía de telecomunicaciones iraní. Tiene 3150 observaciones y 13 columnas.
https://archive.ics.uci.edu/dataset/563/iranian+churn

**Métricas:**

La métrica principal para evaluar los modelos fue el recall al suponer que es más costoso no detectar un cliente que deja el servicio que detectar un cliente que puede dejar el servicio de forma errónea. La suposición se toma porque no se sabe el costo real del abandono o retención del cliente. 

**Modelos utilizados:**

-	Logistic Regression
-	Random Forest
-	XGBoost
-	Feedforward Neural Network (FNN)
-	
**Tecnologías utilizadas:**
 	
-	Python
-	Pandas
-	NumPy
-	PyTorch
-	Streamlit
-	GitHub
-	Matplotlib
-	
**Archivos principales:**

-	Churn_telecomunicaciones.py: aplicación Streamlit desplegada
-	Predicción de Churn de telecomunicaciones con modelos clásicos de machine learning y redes neuronales.ipynb: análisis completo y desarrollo del modelo.

**Resultados:**

**FNN:**

<img width="355" height="78" alt="image" src="https://github.com/user-attachments/assets/d829580d-ec04-40c6-b232-1a832dd4dca6" />

**XGBoost:**

<img width="304" height="52" alt="image" src="https://github.com/user-attachments/assets/cf9a82bb-24e4-4532-8c43-03dd81da03fe" />

**Conclusiones:**

-	El mejor modelo de machine learning clásico es XGBoost sin usar smote, con un umbral de 0.45 mejorando el recall a 0.97.
-	En el modelo de XGBoost se prescindió de dos columnas sin afectar las métricas.
-	La red neuronal alcanzo un 0.95 sin smote y un 0.86 con smote.
-	Tanto la red neuronal sin smote como el modelo de XGBoost son perfectamente válidos para ser usados en la aplicación. XGBoost tiene más recall pero FNN tiene un mayor F1.
-	Se considera XGBoost para ser usado en la aplicación por haber alcanzado el mayor recall.

**Limitaciones:**
 	
-	Los modelos tomaron el recall como la métrica principal por desconocer el costo real de los falsos negativos y positivos.
-	Por limitaciones de hardware los GridSearch no fueron demasiado exhaustivos y en el caso de las redes neuronales no se exploró un mayor número de epoch.

**Aplicación descargable:**

Desde este link se puede acceder a una app para generar predicciones con el modelo ganador: https://prediccion-de-churn-sobre-datos-historicos-de-telecomunicacion.streamlit.app/

