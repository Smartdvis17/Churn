# Predicción de Abandono de Clientes (Churn) — Telco Customer Dataset

---

## 1. Autores

|Diego Orjuela - Alex Rodriguez - Ana Salcedo| 


**Repositorio:** [Smartdvis17/Churn](https://github.com/Smartdvis17/Churn)  
**Curso:** Aprendizaje Automático I — Especialización en Analítica y Ciencia de Datos  
**Universidad de Antioquia**

---

## 2. Descripción del Dataset

**Nombre:** Telco Customer Churn — IBM Watson Analytics 
**Fuente:** [Kaggle — alfathterry](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)  
**Licencia:** CC0 — Dominio Público  
**Naturaleza:** Datos sintéticos con patrones estadísticos reales  

El dataset contiene información de clientes de una empresa de telecomunicaciones ficticia operando en California, EE.UU., durante el tercer trimestre. Fue diseñado originalmente por IBM para demostrar capacidades analíticas en la plataforma Watson Analytics.

### Dimensiones

| Partición | Registros | Variables |
|---|---|---|
| Dataset original (`telco.csv`) | 7,043 | 50 |
| Dataset de entrenamiento (`telco_Prep.csv`) | 6,326 | 41 |
| Dataset de prueba de producción (`telco_Prue.csv`) | 705 | 41 |

### Categorías de variables

| Categoría | Ejemplos |
|---|---|
| Identificación | `Customer ID`, `Quarter` |
| Información demográfica | `Gender`, `Age`, `Senior Citizen`, `Married`, `Dependents` |
| Información geográfica | `City`, `Zip Code`, `Latitude`, `Longitude`, `Population` |
| Fidelización | `Tenure in Months`, `Number of Referrals`, `Offer` |
| Servicios de telefonía | `Phone Service`, `Multiple Lines` |
| Servicios de internet | `Internet Type`, `Online Security`, `Online Backup`, `Streaming TV/Movies/Music` |
| Facturación y pagos | `Contract`, `Monthly Charge`, `Total Charges`, `Payment Method` |
| Satisfacción | `Satisfaction Score`, `CLTV`, `Churn Score` |
| **Variable objetivo** | **`Churn Label`** (Yes/No) — desbalance: 73% No Churn / 27% Churn |

---

## 3. Objetivo a Desarrollar

Construir un **modelo de clasificación** capaz de predecir si un cliente de telecomunicaciones abandonará la compañía (Churn Label = Yes), usando variables demográficas, de contrato y de uso de servicios, excluyendo variables de fuga como (Churn Score, CLTV, Satisfaction Score, Customer Status, Total Revenue).

El objetivo operativo del modelo es **Obtener la mejor combinacion entre el Recall y f1_score de la clase Churn**: detectar el porcentaje mas optimo de posibles abandonos de clientes que efectivamente se van, de tal manera que podamos implementar acciones de retención para evitar la fuga potencial de clientes.

---

## 4. Resumen del Proceso Realizado

```
Dataset bruto (7,043 registros)
         │
         ▼
01. Preparación de datos (notebook 01)
    ├── Eliminación de columnas irrelevantes (Country, City, Churn Category, etc.)
    ├── Imputación de nulos (Offer → "No offer", Internet Type → "No internet")
    ├── Detección y eliminación de outliers (LOF, 12 registros)
    └── División: 90% entrenamiento (6,326) │ 10% prueba de producción (705)
         │
         ▼
02. Regresión Logística (notebook 02)
    ├── 3 solvers: L-BFGS, Liblinear (L1), SAGA (ElasticNet)
    ├── Modelos base (sin/con class_weight="balanced")
    ├── GridSearch métrica: scoring={'recall','f1'}, 
    └── Oversampling evaluado pero descartado (sin mejora significativa en producción)
         │
         ▼
03. Random Forest (notebook 03)
    ├── Modelos base (sin/con class_weight="balanced")
    ├── Grid Search OOB combinado: score = 0.5×Recall_OOB + 0.5×F1_OOB
    │   90 combinaciones - thresholds [0.3, 0.4, 0.5] — 
    └── Oversampling evaluado pero descartado (sobreajuste ligero, sin mejora relevante)
         │
         ▼
04. Evaluación entre modelos (notebook 04)
    ├── Carga  de modelos exportados (.pkl) desde la API de GitHub
    ├── Métricas: Accuracy, Recall, F1, AUC — ordenadas por Recall y f1 score
    ├── Matrices de confusión y curvas ROC comparativas
    └── Modelo ganador: lr_saga_base (RL) — Recall=0.8619, F1=0.7027 en producción
```

---

## 5. Desarrollo de Experimentos

### 5.1 Preparación de Datos (`01_Preparacion_Clasificacion_Churn.ipynb`)

#### Carga del dataset
- Dataset original: 7,043 registros, 50 columnas, sin valores nulos en variables numéricas.
- Variables categóricas con nulos: Offer (3,877) y Internet Type (1,526).

#### Limpieza de variables
Se eliminaron variables que no aportan al modelo o que introducen ruido:

Variable eliminada 

-Churn Category, Churn Reason 
-Customer ID, Country, State, City 
-Quarter, Under 30, Dependents 

#### Imputación de nulos
Los valores NaN no eran inconsistencias sino comportamientos reales del negocio:

| Variable | Valores imputados |
| Offer | No offer 
| Internet Type | `No internet | 

#### Detección de outliers — LOF 
- lo aplicamos sobre variables numéricas de comportamiento (excluyendo coordenadas geográficas y CLTV), se utilizo comparacion con 13 vecinos cercanos para la deteccion de outliers, `n_neighbors=13`, `contamination='auto'`. se obtuvo como resultado la eliminacion de 12 registros atípicos eliminados 

#### División del dataset
Dividimos el dataset en 2 conjuntos:
- **`telco_Prep.csv`** (90% = 6,326 registros):  datos utlizados en el entrenamiento y validación de modelos.
- **`telco_Prue.csv`** (10% = 705 registros): utilizado para prueba final independiente.

---

### 5.2 Creación de Modelos

#### Preparación para el modelado (Regresión Logística - Random Forest)
1. **One-Hot Encoding** lo aplicamos sobre variables categóricas 
2. **Exclusión de variables de fuga**: Churn Score, CLTV, Total Revenue, Customer Status, Satisfaction Score, variables geográficas.
3. **División train/test**: 80% entrenamiento/20% prueba  random_state=123.
4. **Escalado**: Aplicamos MinMaxScaler  sobre las 11 variables numéricas 
---

#### Modelo de Regresión Logística (`02_Regresion_Logistica_Churn.ipynb`)
entrenamos los modelo con 3 solvers (L-BFGS, Liblinear (L1), SAGA (ElasticNet)) y evaluamos con diferentes estrategias de balance:


**Estrategias de balanceo:**
- Base Balanced: 
- Base None: aplicamos el modelo con los mismos parametros sobre la base desbalanceada para determinar el desempeño del modelo base sin balanceo.
- **GridSearch Balanced**: inicialmente enfocamos nuestro modelo hacia la optimizacion del recall, sin enmbargo tomamos la decision de encontrar una solucion mas adecuada que nos permitiera un balance entre Recall y F1, para ello, configuramos el modelo para obtener un modelo más estable entre sensibilidad (Recall) y rendimiento global (F1).
- **Oversampling**: RandomOverSampler lo evaluamos durante el proceso de creacion de nuestros modelos pero no encontramos mejorar significativas en el desempeño del modelo por lo que decidimos descartarlo. 

> Pasamos de priorizar únicamente Recall a a considerar una evaluación mas equilibrada entre (Recall + F1), logrando mayor estabilidad sin sacrificar la capacidad de detección. 
El modelo ganador de Regresión Logística es el `lr_saga_base` con Recall=0.8619, F1=0.7027 en producción.



---

#### Modelo Random Forest (`03_Random_Forest_Churn.ipynb`)

**Modelos base entrenados:**

| Modelo | Train Acc | Recall (val) | F1 (val) | AUC (val) |
|---|---|---|---|---|
| Base sin balanceo | 0.8263 | 0.4819 | 0.5850 | 0.8847 |
| Base con balanceo | 0.7964 | 0.8434 | 0.6675 | 0.8858 |


**Grid Search OOB — Recall + F1**

| Hiperparámetro | Valores probados |
|---|---|
| n_estimators | 100, 150, 200 |
| max_features | sqrt, log2, 5, 7, 9 |
| max_depth | 5, 10, 15 |
| criterion | gini, entropy |

en la optimizacion de nuestro modelo de random forest con GridSearch decidimos buscar el mejor modelo con la combinacion de Recall y F1.

Usamos oob por que al intentar con cv nos tomo mucho tiempo de ejecucion y oob nos funciono mejor tambien intentamos añadir los siguientes parametros      'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], para determinar si nos podian reflejar una optimizacion del modelo sin embargo el tiempo de respuesta no fue optimo (+7 min) y decidimos desestimar estos datos y trabajar solo con los siguientes parametros con los que tardamos alrededor de 3 min en ejecucion.
      'n_estimators': [100, 150, 200],
      'max_features': ['sqrt', 'log2', 5, 7, 9],
      'max_depth': [5, 10, 15],  
      'criterion': ['gini', 'entropy'],


aqui vemos una comparacion del cambio en nuestros hyperámetros con respecto a la version inicial de nuestro modelo de Random Forest.
| Versión inicial (solo Recall) | Maximizar Recall | criterion=entropy, depth=5, max_features=9, n=200 | 0.8618 |
| Versión final (Recall + F1) | Score combinado | criterion=entropy, depth=5, max_features=log2, n=200 | 0.8507 |

> Aunque el Recall OOB es ligeramente menor en la versión combinada, el modelo es más equilibrado entre sensibilidad y precisión global, lo que se traduce en mayor equilibrio entre las 2 metricas.

---

### 5.3 Evaluación de Modelos (`04_Evaluacion_modelos.ipynb`)

**Resultados Regresión Logística (ordenados por Recall, `telco_Prue.csv` — 705 registros, 181 Churn):**

Despues de realizar la implementacion de pruebas con los datos reservaod identificacion que nuestro modelo mas equilibrado es el `lr_saga_base` con Recall=0.8619, F1=0.7027.

| Modelo | Accuracy | Recall | F1 | AUC |
|---|---|---|---|---|
| **lr_saga_base** | **0.8128** | **0.8619** | **0.7027** | **0.8288** |
| lr_lbfgs_base | 0.8113 | 0.8619 | 0.7011 | 0.8279 |
| lr_liblinear_base | 0.8113 | 0.8564 | 0.6998 | 0.8261 |
| lr_liblinear_opt_balanced | 0.8071 | 0.8508 | 0.6937 | 0.8214 |
| lr_saga_opt_balanced | 0.8071 | 0.8508 | 0.6937 | 0.8214 |
| lr_lbfgs_opt_balanced | 0.8057 | 0.8508 | 0.6921 | 0.8205 |
| lr_lbfgs_base_none | 0.8511 | 0.6630 | 0.6957 | 0.7895 |
| lr_saga_base_none | 0.8511 | 0.6630 | 0.6957 | 0.7895 |
| lr_liblinear_base_none | 0.8496 | 0.6630 | 0.6936 | 0.7886 |


**Resultados Random Forest (ordenados por Recall):**

| Modelo | Accuracy | Recall | F1 | AUC | Churn detectados | Churn perdidos |
|---|---|---|---|---|---|---|
| RForest_recall_bal | 0.7943 | 0.8343 | 0.6756 | 0.8074 | 151 | 30 |
| RForest_base_bal | 0.7972 | 0.8287 | 0.6772 | 0.8075 | 150 | 31 |
| RForest_base_unbal | 0.8043 | 0.3978 | 0.5106 | 0.6712 | 72 | 109 |

observamos que el modelo `RForest_recall_bal` es el ganador de la competencia con Recall=0.8343, F1=0.6756, sin embargo el modelo de regresion logistica demuestra un mayor balance entre el recall y el F1.

**Herramientas utilizadas:**
- `scikit-learn`: modelos, métricas, preprocesado.
- `imbalanced-learn`: RandomOverSampler (exploración, no exportado en versión final).
- `joblib`: serialización de modelos.
- `matplotlib` / `seaborn`: visualizaciones.

---

## 6. Conclusiones

1. Es importante mantener un equilibrio entre las 2 metricas de evaluacion de tal manera que la combinacion de Recall y F1 sea la mejor. evitando enfocarse únicamente en detectar churners y descuidar la calidad general de las predicciones.

2. El balanceo de las clases es importante para garantizar un buen desempeño en el rendimiento del modelo.

3. El uso de OOB redujo significativamente el costo computacional frente a Cross-Validation (270 evaluaciones en ~3 min vs ~450 con CV en ~7 min).

5. Los modelos orientados a maximizar Recall generan más alarmas falsas (falsos positivos). El modelo ganador (`lr_saga_base`) detecta ~156 de 181 churners en producción.


---

## 7. Referencias

- **Dataset:** Telco Customer Churn. Kaggle — alfathterry. [https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)

