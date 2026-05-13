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

Construir un **modelo de clasificación binaria** capaz de predecir si un cliente de telecomunicaciones abandonará la compañía (`Churn Label = Yes`), usando variables demográficas, de contrato y de uso de servicios, **sin incluir variables de fuga** (Churn Score, CLTV, Satisfaction Score, Customer Status, Total Revenue).

El objetivo operativo del modelo es **maximizar el Recall de la clase Churn**: detectar el mayor porcentaje posible de clientes que efectivamente se van, de tal manera que podamos implementar acciones de retención para evitar la fuga potencial de clientes.

---

## 4. Resumen del Proceso Realizado

```
Dataset bruto (7,043 registros)
         │
         ▼
01. Preparación de datos
    ├── Eliminación de columnas irrelevantes (Country, City, Churn Category, etc.)
    ├── Imputación semántica de nulos (Offer → "No offer", Internet Type → "No internet")
    ├── Detección y eliminación de outliers multivariados (LOF, 12 registros)
    └── División: 90% entrenamiento (6,326) │ 10% prueba de producción (705)
         │
         ▼
02. Regresión Logística (notebook 02)
    ├── 3 solvers: L-BFGS, Liblinear (L1), SAGA (ElasticNet)
    ├── Estrategias: base, base-None, GridSearch (None/Balanced), Oversampling
    └── GridSearchCV con scoring='recall' — criterio alineado al objetivo operativo
         │
         ▼
03. Random Forest (notebook 03)
    ├── Modelos base (sin/con class_weight="balanced")
    ├── Grid Search OOB-Recall (90 combinaciones): Recall calculado desde
    │   oob_decision_function_ sobre y_train — sin tocar X_test
    └── Modelo con Oversampling (RandomOverSampler)
         │
         ▼
04. Evaluación entre modelos (notebook 04)
    ├── Carga y comparación de todos los modelos exportados (.pkl)
    ├── Métricas: Accuracy, Recall, F1, AUC — ordenadas por Recall
    ├── Matrices de confusión y curvas ROC
    └── Modelo ganador: RF Recall Balanceado — mayor Recall con control de overfitting
```

---

## 5. Desarrollo de Experimentos

### 5.1 Preparación de Datos (`01_Preparacion_Clasificacion_Churn.ipynb`)

#### Carga e inspección
- Dataset original: 7,043 registros, 50 columnas, sin valores nulos en variables numéricas.
- Variables categóricas con nulos: `Offer` (3,877) y `Internet Type` (1,526).

#### Limpieza de variables
Se eliminaron variables que no aportan a la predicción de churn o que introducen ruido:

| Variable eliminada | Motivo |
|---|---|
| `Churn Category`, `Churn Reason` | Solo están diligenciadas para clientes que abandonan (fuga potencial y sesgo) |
| `Customer ID`, `Country`, `State`, `City` | No aportan comportamiento de negocio |
| `Quarter`, `Under 30`, `Dependents` | Redundantes con otras variables |

#### Imputación de nulos
Los valores `NaN` no eran inconsistencias sino comportamientos reales del negocio:

| Variable | Valor imputado | Justificación |
|---|---|---|
| `Offer` | `"No offer"` | El cliente no tiene oferta activa |
| `Internet Type` | `"No internet"` | El cliente no tiene internet contratado |

#### Detección de outliers — LOF (Local Outlier Factor)
- Aplicado sobre 15 variables numéricas de comportamiento (excluyendo coordenadas geográficas y CLTV).
- Configuración: `n_neighbors=13`, `contamination='auto'`.
- Resultado: **12 registros atípicos eliminados** (de 6,338 → 6,326).

#### División del dataset
División manual previa al proceso de modelado:
- **`telco_Prep.csv`** (90% = 6,326 registros): entrenamiento y validación de modelos.
- **`telco_Prue.csv`** (10% = 705 registros): prueba de producción independiente.

---

### 5.2 Creación de Modelos

#### Preparación para el modelado (común a ambos modelos)
1. **One-Hot Encoding** sobre variables categóricas (`pd.get_dummies`, `drop_first=True`).
2. **Exclusión de variables de fuga**: `Churn Score`, `CLTV`, `Total Revenue`, `Customer Status`, `Satisfaction Score`, variables geográficas.
3. **División train/test**: 80% entrenamiento (5,060) / 20% prueba (1,266), `random_state=123`.
4. **Escalado**: `MinMaxScaler` aplicado sobre las 11 variables numéricas del conjunto de entrenamiento.

---

#### Modelo de Regresión Logística (`02_Regresion_Logistica_Churn.ipynb`)

Se entrenaron modelos con **3 solvers** y diferentes estrategias de balance:

| Solver | Regularización | Notas |
|---|---|---|
| L-BFGS | L2 (defecto) | Convergencia robusta |
| Liblinear | L1 (lasso) | Selección de variables |
| SAGA | ElasticNet (l1_ratio=0.2) | Combinación L1+L2 |

**Estrategias evaluadas:**
- **Base Balanced**: `class_weight="balanced"` — penaliza errores en la clase Churn.
- **Base None**: sin corrección de clases — punto de referencia desbalanceado.
- **GridSearch — Config A (None)**: búsqueda de `C` óptimo con `scoring='recall'`, sin balance.
- **GridSearch — Config B (Balanced)**: búsqueda de `C` óptimo con `scoring='recall'` y `class_weight='balanced'`.
- **Oversampling**: `RandomOverSampler` sobre el training set.

> Ambas configuraciones de GridSearch usan `scoring='recall'` en coherencia con el objetivo operativo del proyecto.

**Resultados comparativos — Regresión Logística:**

| Configuración | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| L-BFGS base (Balanced) | 0.8057 | 0.5885 | 0.8614 | 0.6993 | ~0.91 |
| L-BFGS base (None) | 0.8523 | 0.7425 | 0.6687 | 0.7036 | ~0.91 |
| L-BFGS opt-None (scoring=recall) | 0.8523 | 0.7377 | 0.6777 | 0.7064 | ~0.91 |
| L-BFGS opt-Balanced (scoring=recall) | 0.8057 | 0.5885 | 0.8614 | 0.6993 | ~0.91 |
| **Liblinear opt-Balanced (scoring=recall)** | **0.7970** | **0.5737** | **0.8795** | **0.6944** | ~0.91 |
| SAGA opt-Balanced (scoring=recall) | 0.8057 | 0.5885 | 0.8614 | 0.6993 | ~0.91 |

> El mejor modelo de RL por Recall es **Liblinear opt-Balanced** con Recall = 0.8795.  
> Todos los modelos de RL mostraron **sin overfitting** (diferencia Train-Test ≤ 0.01).

---

#### Modelo Random Forest (`03_Random_Forest_Churn.ipynb`)

**Modelos base entrenados:**

| Modelo | Train Acc | F1 Test | AUC |
|---|---|---|---|
| Base sin balanceo | 0.8263 | 0.5850 | 0.8847 |
| Base con balanceo | 0.7964 | 0.6675 | 0.8858 |

**Grid Search OOB — Recall — espacio de búsqueda:**

| Hiperparámetro | Valores probados |
|---|---|
| `n_estimators` | 100, 150, 200 |
| `max_features` | `sqrt`, `log2`, 5, 7, 9 |
| `max_depth` | 5, 10, 15 |
| `criterion` | `gini`, `entropy` |

Total: **90 combinaciones** evaluadas. Para cada combinación se calcula el **Recall OOB** directamente desde `oob_decision_function_` sobre `y_train`.

**¿Por qué OOB en lugar de Cross-Validation?**

| Criterio | CV (k=5) | OOB |
|---|---|---|
| Forests entrenados | 90 × 5 = 450 | 90 × 2 = 180 |
| Usa X_test | No | No |
| Natural para RF | No | Sí (bootstrap integrado) |
| Métrica optimizada | Recall | Recall OOB |

**Criterio de selección:** mayor `Recall OOB` con `gap (train_acc − oob_acc) < 0.07` como control de overfitting.

**Oversampling con RandomOverSampler:**
- Distribución antes: No Churn 3,712 / Churn 1,348.
- Distribución después: No Churn 3,712 / Churn 3,712 (balanceo 1:1).

**Resultados comparativos — Random Forest (ordenados por Recall):**

| Modelo | Accuracy | Precision | Recall | F1 | AUC | Churn detectados | Churn perdidos |
|---|---|---|---|---|---|---|---|
| Base con balanceo | 0.7796 | 0.5523 | 0.8434 | 0.6675 | 0.8858 | 280 | 52 |
| RF + Oversampling | 0.7994 | 0.5908 | 0.8012 | 0.6801 | 0.8864 | 270 | 67 |
| **Recall Balanceado (OOB-Recall)** | **0.7796** | **0.5511** | **0.8614** | **0.6722** | **0.8896** | **286** | **46** |
| Recall sin balanceo (OOB-Recall) | 0.8215 | 0.7366 | 0.4970 | 0.5935 | 0.8888 | 165 | 167 |
| Base sin balanceo | 0.8207 | 0.7442 | 0.4819 | 0.5850 | 0.8847 | 160 | 172 |

**Análisis de overfitting:**

| Modelo | Train Acc | Test Acc | Gap | Nivel |
|---|---|---|---|---|
| Base sin balanceo | 0.8263 | 0.8207 | 0.0056 | Sin overfitting |
| Base con balanceo | 0.7964 | 0.7796 | 0.0168 | Overfitting leve |
| Optimizado con balanceo (OOB-Recall) | 0.7909 | 0.7796 | 0.0113 | Overfitting leve |
| Optimizado sin balanceo (OOB-Recall) | 0.8261 | 0.8215 | 0.0046 | Sin overfitting |
| RF + Oversampling | 0.9181 | 0.7994 | 0.1187 | Overfitting alto |

---

### 5.3 Evaluación de Modelos (`04_Evaluacion_modelos.ipynb`)

Se cargaron dinámicamente todos los modelos exportados en formato `.pkl` desde el repositorio de GitHub y se evaluaron sobre el conjunto de prueba de producción (`telco_Prue.csv`), que nunca fue visto durante el entrenamiento.

**Proceso de evaluación:**
1. Descarga automática de modelos `.pkl` desde la API de GitHub.
2. Transformación del conjunto de prueba: One-Hot Encoding, exclusión de variables de fuga, escalado con el scaler exportado (`minmaxFull_Telco.pkl`).
3. Predicción y cálculo de métricas: Accuracy, Recall, F1, AUC.
4. **Ranking por Recall** — criterio principal alineado al objetivo operativo.
5. Visualización: matrices de confusión y curvas ROC comparativas.

**Modelos evaluados:** 20 modelos (5 Random Forest + 15 Regresión Logística con 3 solvers × 5 estrategias.

**Herramientas utilizadas:**
- `scikit-learn`: modelos, métricas, preprocesado.
- `imbalanced-learn`: RandomOverSampler.
- `joblib`: serialización de modelos.
- `matplotlib` / `seaborn`: visualizaciones.

---

## 6. Conclusiones

1. Optimizar con `scoring='recall'` en GridSearchCV (RL) y con Recall OOB desde `oob_decision_function_` (RF) garantiza que la búsqueda de hiperparámetros busque directamente el objetivo del negocio: detectar el mayor número posible de clientes en riesgo.

2. A diferencia de Cross-Validation (que requiere entrenar k bosques adicionales), el OOB aprovecha el bootstrap integrado del algoritmo. Calcula Recall desde `oob_decision_function_` sobre `y_train` y ofrece una estimación evitando data leakage en la selección de hiperparámetros.

3. RF captura mejor las relaciones no lineales del dataset.

4. El desbalance original (73% / 27%) hace que los modelos sin corrección tiendan a ignorar la clase Churn. 

5. Los modelos orientados a maximizar Recall generan más alarmas falsas. 

6. Los modelos optimizados de RF presentan mayor brecha entre accuracy de entrenamiento y prueba. 

7. Variables como `Churn Score`, `CLTV`, `Satisfaction Score` y `Customer Status` son calculadas post-evento y no estarían disponibles en tiempo real. Su inclusión inflaría artificialmente las métricas del modelo.

---

## 7. Referencias

- **Dataset:** Telco Customer Churn. Kaggle — alfathterry. [https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)

---

## Estructura del Repositorio

```
Churn/
├── datasets/
│   ├── telco.csv                     # Dataset original (7,043 registros)
│   ├── telco_Prep.csv                # Dataset de entrenamiento (6,326 registros post-limpieza)
│   └── telco_Prue.csv                # Dataset de prueba de producción (705 registros)
├── modelos/
│   ├── clasificacion/
│   │   ├── RForest_base_unbal.pkl    # RF base sin balanceo
│   │   ├── RForest_base_bal.pkl      # RF base con class_weight='balanced'
│   │   ├── RForest_recall_bal.pkl    # RF optimizado OOB-Recall con balanceo  ← modelo principal
│   │   ├── RForest_recall_unbal.pkl  # RF optimizado OOB-Recall sin balanceo
│   │   ├── RForest_oversampling.pkl  # RF con RandomOverSampler
│   │   ├── lr_lbfgs_*.pkl            # Modelos RL solver L-BFGS (5 variantes)
│   │   ├── lr_liblinear_*.pkl        # Modelos RL solver Liblinear (5 variantes)
│   │   └── lr_saga_*.pkl             # Modelos RL solver SAGA (5 variantes)
│   └── scaler/
│       └── minmaxFull_Telco.pkl      # MinMaxScaler ajustado sobre X_train
├── utils/
│   └── funciones.py                  # Funciones auxiliares (multiple_plot, plot_roc_curve)
├── 01_Preparacion_Clasificacion_Churn.ipynb
├── 02_Regresion_Logistica_Churn.ipynb
├── 03_Random_Forest_Churn.ipynb
├── 04_Evaluacion_modelos.ipynb
└── README.md
```
