# Se importan las librerias para el template y los renders
from django.shortcuts import render
import json

# Libreria para operaciones matemáticas
import numpy as np
# Libreria para manejo de datos
import pandas as pd

# Libreria para transformacion de datos
from sklearn.preprocessing import LabelEncoder
# Libreria para calcular la significancia estadística y el coeficiente de correlación
from scipy import stats
# Libreria para separar los datos
from sklearn.model_selection import train_test_split
# Libreria para Regresión Lineal
from sklearn.linear_model import LinearRegression
# Librerías para métricas
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Ignorar Warning
import warnings
warnings.simplefilter('ignore')
import os
# -----------------------------------

def main(request):
    # *** Plantilla inicial sin datos ***
    return render(request, 'index.html', context={})


def prediccion(request):

    # Se cargan los datos de entrada
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '..', 'data', 'DatosValoraciones-Proyecto.csv')

    data = pd.read_csv(csv_path, sep=';')
    data.drop(['PersonaqueleAyuda', 'AsisteaRehabilitacion'], axis=1, inplace=True)

    # Se cambia género a valor numerico
    le = LabelEncoder()

    data['CondicionDiscapacidad'] = le.fit_transform( data['CondicionDiscapacidad'])
    data['Sexo'] = le.fit_transform( data['Sexo'])
    data['MomentoCursoVida'] = le.fit_transform( data['MomentoCursoVida'])
    data['Dis_fisica'] = le.fit_transform( data['Dis_fisica'])
    data['Dis_Visual'] = le.fit_transform( data['Dis_Visual'])
    data['Dis_Auditiva'] = le.fit_transform( data['Dis_Auditiva'])
    data['Dis_Intelectual'] = le.fit_transform( data['Dis_Intelectual'])
    data['Dis_Psicosocial'] = le.fit_transform( data['Dis_Psicosocial'])
    data['Dis_Sordoceguera'] = le.fit_transform( data['Dis_Sordoceguera'])
    data['Dis_Multiple'] = le.fit_transform( data['Dis_Multiple'])
    data['CausaDeficiencia'] = le.fit_transform( data['CausaDeficiencia'])
    data['IdentidaddeGenero'] = le.fit_transform( data['IdentidaddeGenero'])
    data['OrientacionSexual'] = le.fit_transform( data['OrientacionSexual'])
    data['HaEstadoProcesosdeRehabilitacion'] = le.fit_transform( data['HaEstadoProcesosdeRehabilitacion'])
    data['SuMunicipioTieneServiciodeRehabilitacion'] = le.fit_transform( data['SuMunicipioTieneServiciodeRehabilitacion'])
    data['UtilizaProductosApoyo'] = le.fit_transform( data['UtilizaProductosApoyo'])
    data['LeeyEscribe'] = le.fit_transform( data['LeeyEscribe'])
    data['AsisteaInstitucionEducativa'] = le.fit_transform( data['AsisteaInstitucionEducativa'])
    data['NivelEducativo'] = le.fit_transform( data['NivelEducativo'])
    data['Trabaja'] = le.fit_transform( data['Trabaja'])
    data['FuenteIngresos'] = le.fit_transform( data['FuenteIngresos'])
    data['IngresoMensualPromedio'] = le.fit_transform( data['IngresoMensualPromedio'])
    data['RequiereAyudadeOtraPersona'] = le.fit_transform( data['RequiereAyudadeOtraPersona'])
    data['PersonaqueledaApoyoyRespaldo'] = le.fit_transform( data['PersonaqueledaApoyoyRespaldo'])
    data['BarrerasFisicasVivienda'] = le.fit_transform( data['BarrerasFisicasVivienda'])
    data['BarrerasFisicasEspaciopúblico'] = le.fit_transform( data['BarrerasFisicasEspaciopúblico'])
    data['BarrerasFisicasTransportepúblico'] = le.fit_transform( data['BarrerasFisicasTransportepúblico'])
    data['BarrerasFisicasEdificacionespúblicasprivadas'] = le.fit_transform( data['BarrerasFisicasEdificacionespúblicasprivadas'])
    data['ActitudesNegativasenInteraccion1'] = le.fit_transform( data['ActitudesNegativasenInteraccion1'])
    data['ActitudesNegativasenInteraccion2'] = le.fit_transform( data['ActitudesNegativasenInteraccion2'])
    data['ActitudesNegativasenInteraccion3'] = le.fit_transform( data['ActitudesNegativasenInteraccion3'])
    data['ActitudesNegativasenInteraccion4'] = le.fit_transform( data['ActitudesNegativasenInteraccion4'])
    data['ActitudesNegativasenInteraccion5'] = le.fit_transform( data['ActitudesNegativasenInteraccion5'])
    data['ActitudesNegativasenInteraccion6'] = le.fit_transform( data['ActitudesNegativasenInteraccion6'])
    data['ActitudesNegativasenInteraccion7'] = le.fit_transform( data['ActitudesNegativasenInteraccion7'])
    data['ActitudesNegativasenInteraccion8'] = le.fit_transform( data['ActitudesNegativasenInteraccion8'])

    # Llenar todos los NaN con cero
    data.fillna(0, inplace=True)

    ##PRIMER MODELO - Predicción de Discapacidad Global basada en Discapacidad Intelectual
    # Variable independiente como DataFrame
    X = data[['Niv_DisIntelectual']]
    y = data['Niv_DisGlobal']

    # Separar datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

    # Se define el algoritmo de predicción
    lr1 = LinearRegression()

    # Se entrena el modelo
    lr1.fit(X_train, y_train)

    # Se genera la predicción
    prediction1 = lr1.predict(X_test)

    #################################################################################################################

    ##SEGUNDO MODELO - Predicción de Sordoceguera basada en Discapacidad Intelectual

    # Variable independiente como DataFrame
    X2 = data[['Niv_DisIntelectual']]
    y2 = data['Niv_DisSordoceguera']

    # Separar datos
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.8, random_state=1)

    # Se define el algoritmo de predicción
    lr2 = LinearRegression()

    # Se entrena el modelo
    lr2.fit(X_train2, y_train2)

    # Se genera la predicción
    prediction2 = lr2.predict(X_test2)

    # Se convierten los valores a listas y se redondean las predicciones para mejor visualización
    valores_reales = [round(val, 3) for val in y_test.tolist()]
    valores_predichos = [round(val, 3) for val in prediction1.tolist()]
    valores_reales_modelo2 = [round(val, 3) for val in y_test2.tolist()]
    valores_predichos_modelo2 = [round(val, 3) for val in prediction2.tolist()]

    # Calcular errores individuales para gráfica de barras
    errores_modelo1 = [abs(real - pred) for real, pred in zip(valores_reales, valores_predichos)]
    errores_modelo2 = [abs(real - pred) for real, pred in zip(valores_reales_modelo2, valores_predichos_modelo2)]

    # **MÉTRICAS DEL MODELO 1**
    r2_modelo1 = round(r2_score(y_test, prediction1), 4)
    mae_modelo1 = round(mean_absolute_error(y_test, prediction1), 4)
    mse_modelo1 = round(mean_squared_error(y_test, prediction1), 4)
    rmse_modelo1 = round(np.sqrt(mse_modelo1), 4)

    # Características del Modelo 1 (y real)
    media_modelo1 = round(np.mean(y_test), 4)
    std_modelo1 = round(np.std(y_test), 4)
    max_modelo1 = round(np.max(y_test), 4)
    min_modelo1 = round(np.min(y_test), 4)
    unicos_modelo1 = len(np.unique(y_test))

    # **MÉTRICAS DEL MODELO 2**
    r2_modelo2 = round(r2_score(y_test2, prediction2), 4)
    mae_modelo2 = round(mean_absolute_error(y_test2, prediction2), 4)
    mse_modelo2 = round(mean_squared_error(y_test2, prediction2), 4)
    rmse_modelo2 = round(np.sqrt(mse_modelo2), 4)

    # Características del Modelo 2 (y real)
    media_modelo2 = round(np.mean(y_test2), 4)
    std_modelo2 = round(np.std(y_test2), 4)
    max_modelo2 = round(np.max(y_test2), 4)
    min_modelo2 = round(np.min(y_test2), 4)
    unicos_modelo2 = len(np.unique(y_test2))

    # Se envian los valores al contexto para renderizar
    context = {
        # Datos para gráficas
        "valores_reales": json.dumps(valores_reales),
        "valores_predichos": json.dumps(valores_predichos),
        "valores_reales_modelo2": json.dumps(valores_reales_modelo2),
        "valores_predichos_modelo2": json.dumps(valores_predichos_modelo2),
        
        # Errores para gráfica de barras
        "errores_modelo1": json.dumps(errores_modelo1),
        "errores_modelo2": json.dumps(errores_modelo2),
        
        # Métricas Modelo 1
        "r2_modelo1": r2_modelo1,
        "mae_modelo1": mae_modelo1,
        "mse_modelo1": mse_modelo1,
        "rmse_modelo1": rmse_modelo1,
        
        # Características Modelo 1
        "media_modelo1": media_modelo1,
        "std_modelo1": std_modelo1,
        "max_modelo1": max_modelo1,
        "min_modelo1": min_modelo1,
        "unicos_modelo1": unicos_modelo1,
        
        # Métricas Modelo 2
        "r2_modelo2": r2_modelo2,
        "mae_modelo2": mae_modelo2,
        "mse_modelo2": mse_modelo2,
        "rmse_modelo2": rmse_modelo2,
        
        # Características Modelo 2
        "media_modelo2": media_modelo2,
        "std_modelo2": std_modelo2,
        "max_modelo2": max_modelo2,
        "min_modelo2": min_modelo2,
        "unicos_modelo2": unicos_modelo2,
    }
    
    return render(request, 'index.html', context=context)