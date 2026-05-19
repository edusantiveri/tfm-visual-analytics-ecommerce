# Sistema de Visual Analytics para la exploración, limpieza y análisis de datos orientada a la toma de decisiones 

Este repositorio contiene el artefacto software desarrollado como resultado del Trabajo Fin de Máster (TFM) para el **Máster Universitario en
Análisis y Visualización de Datos Masivos / Visual Analytics & Big Data** de la **Universidad Internacional de La Rioja (UNIR)**.

El sistema es una plataforma interactiva de *Visual Data Wrangling* que unifica el perfilado de calidad de datos (*Data Profiling*) con la exploración analítica de KPIs operativos de comercio electrónico, siguiendo un enfoque metodológico *Design Science Research* (DSR).

---

## 🚀 Acceso Inmediato (Producción)

Para facilitar la evaluación por parte del tribunal y evitar configuraciones locales, la aplicación se encuentra totalmente operativa en la nube a través del siguiente enlace:

👉 **[Accede aquí a la Aplicación en Streamlit Cloud]([https://tu-proyecto-tfm.streamlit.app]https://tfm-visual-analytics-ecommerce-cgemexpthcsw95wwl2nf4f.streamlit.app/])** 

---

## 🛠️ Arquitectura y Funcionalidades Clave

El software implementa un flujo de trabajo estructurado en tres niveles interactivos:
1. **Fase 1 (Normalización Automática):** Adecuación de esquemas tabulares, formateo de nombres a `snake_case` e inferencia segura de tipos de datos (`datetime64` y `float64`).
2. **Fase 2 (Curación Guiada - *Human-in-the-loop*):** Detección interactiva y resolución guiada de registros duplicados, valores ausentes (con estrategias de imputación o filtrado) y aislamiento visual de *outliers* mediante diagramas de caja.
3. **Fase 3 (Auditoría y Análisis):** Visualización de KPIs analíticos (Facturación, Ticket Medio, Volúmenes) tras la limpieza, descarga de un log de auditoría (*Data Lineage*) y exportación del CSV depurado.

---
## 📊 Fuentes de Datos y Reproducibilidad

Los datos utilizados para la fase de pruebas y demostración provienen de fuentes públicas de e-commerce alojadas en **Kaggle**. Debido a las limitaciones de espacio de GitHub para archivos pesados, la distribución se organiza así:
* En la carpeta `project/data/` se encuentran muestras ligeras listas para ejecutar y probar la app localmente de forma inmediata.
* Los datasets completos originales pueden descargarse directamente desde sus fuentes oficiales en Kaggle: [[Enlace 1](https://www.kaggle.com/datasets/jockeroika/ecommerce-data)](URL) | [[Enlace 2](https://www.kaggle.com/datasets/prince7489/e-commerce-sales)](URL) | [[Enlace 3](https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business)](URL) | [[Enlace 4](https://www.kaggle.com/datasets/deepanshuverma0154/sales-dataset-of-ecommerce-electronic-products)](URL) | [[Enlace 5]](https://www.kaggle.com/datasets/abhayayare/e-commerce-dataset?select=ecommerce_dataset)](URL)

---

## 💻 Guía de Instalación y Ejecución en Local

Si desea auditar el código fuente o ejecutar la herramienta de forma local en su equipo, siga estos pasos:

### 1. Prerrequisitos
Asegúrese de tener instalado **Python 3.10 o superior** en su sistema.

### 2. Clonar el repositorio
Abra su terminal y clone este proyecto:
```bash
git clone [[https://github.com/tu-usuario/tu-repositorio-tfm.git](https://github.com/tu-usuario/tu-repositorio-tfm.git)
cd tu-repositorio-tfm/project](https://github.com/edusantiveri/tfm-visual-analytics-ecommerce)
