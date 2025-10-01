import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

class PerceptronSoftmax:
    def __init__(self, tasa_aprendizaje=0.01, iteraciones=100):
        """
        Inicializa el perceptrón con función de activación Softmax

        Parámetros:
        - tasa_aprendizaje: velocidad de ajuste de los pesos
        - iteraciones: número de épocas de entrenamiento (>10)
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None
        self.errores_por_epoca = []
        self.perdida_por_epoca = []
        self.pesos_historia = []
        self.sesgo_historia = []

    def softmax(self, z):
        """
        Función de activación Softmax
        Para clasificación binaria, usamos softmax de 2 clases
        Retorna las probabilidades de cada clase
        """
        # Para evitar overflow, restamos el máximo
        z_exp = np.exp(z - np.max(z))
        return z_exp / np.sum(z_exp)

    def inicializar_pesos(self, n_caracteristicas, n_clases=2):
        """Inicializa los pesos y sesgos para cada clase"""
        # Matriz de pesos: cada columna es para una clase
        self.pesos = np.random.randn(n_caracteristicas, n_clases) * 0.01
        self.sesgo = np.zeros(n_clases)

    def predecir_probabilidades(self, x):
        """Calcula las probabilidades para cada clase usando softmax"""
        z = np.dot(x, self.pesos) + self.sesgo
        return self.softmax(z)

    def predecir(self, x):
        """Realiza una predicción (clase con mayor probabilidad)"""
        probabilidades = self.predecir_probabilidades(x)
        return np.argmax(probabilidades)

    def calcular_perdida(self, X, y):
        """Calcula la pérdida de entropía cruzada"""
        m = len(y)
        perdida_total = 0

        for i in range(m):
            probs = self.predecir_probabilidades(X[i])
            # Evitar log(0)
            prob_clase_correcta = np.clip(probs[y[i]], 1e-10, 1.0)
            perdida_total -= np.log(prob_clase_correcta)

        return perdida_total / m

    def entrenar(self, X, y):
        """
        Entrena el perceptrón con los datos de entrada usando Softmax

        Parámetros:
        - X: matriz de características (n_muestras, n_caracteristicas)
        - y: vector de etiquetas (n_muestras,) con valores 0 o 1
        """
        n_muestras, n_caracteristicas = X.shape
        self.inicializar_pesos(n_caracteristicas)

        print(f"{'='*70}")
        print(f"INICIANDO ENTRENAMIENTO DEL PERCEPTRÓN")
        print(f"{'='*70}")
        print(f"Función de activación: SOFTMAX")
        print(f"Número de iteraciones: {self.iteraciones}")
        print(f"Tasa de aprendizaje: {self.tasa_aprendizaje}")
        print(f"Número de muestras: {n_muestras}")
        print(f"Número de características: {n_caracteristicas}")
        print(f"{'='*70}\n")

        for epoca in range(self.iteraciones):
            errores = 0

            for i in range(n_muestras):
                x_i = X[i]
                y_i = y[i]

                # Forward pass: calcular probabilidades
                probabilidades = self.predecir_probabilidades(x_i)

                # Crear vector one-hot para la clase verdadera
                y_one_hot = np.zeros(2)
                y_one_hot[y_i] = 1

                # Calcular gradiente (diferencia entre predicción y realidad)
                gradiente = probabilidades - y_one_hot

                # Actualizar pesos y sesgo para cada clase
                self.pesos -= self.tasa_aprendizaje * np.outer(x_i, gradiente)
                self.sesgo -= self.tasa_aprendizaje * gradiente

                # Contar errores
                prediccion = np.argmax(probabilidades)
                if prediccion != y_i:
                    errores += 1

            # Calcular pérdida de la época
            perdida = self.calcular_perdida(X, y)

            self.errores_por_epoca.append(errores)
            self.perdida_por_epoca.append(perdida)
            self.pesos_historia.append(self.pesos.copy())
            self.sesgo_historia.append(self.sesgo.copy())

            if epoca < 5 or epoca % 20 == 0 or epoca == self.iteraciones - 1:
                print(f"Época {epoca + 1:3d}/{self.iteraciones} | "
                      f"Errores: {errores:2d} | "
                      f"Pérdida: {perdida:.6f}")

        print(f"\n{'='*70}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"Pesos finales (Clase 0):\n{self.pesos[:, 0]}")
        print(f"Pesos finales (Clase 1):\n{self.pesos[:, 1]}")
        print(f"Sesgo final: {self.sesgo}")
        print(f"Errores en última época: {self.errores_por_epoca[-1]}")
        print(f"Pérdida final: {self.perdida_por_epoca[-1]:.6f}")
        print(f"{'='*70}\n")

    def evaluar(self, X, y, nombres_caracteristicas=None, mostrar_todas=False):
        """Evalúa el perceptrón en un conjunto de datos"""
        print(f"{'='*70}")
        print(f"EVALUACIÓN DEL MODELO")
        print(f"{'='*70}\n")

        predicciones = []
        probabilidades_todas = []

        # Decidir cuántas muestras mostrar en detalle
        n_mostrar = len(X) if mostrar_todas or len(X) <= 20 else 10

        for i, (x, y_real) in enumerate(zip(X, y)):
            probs = self.predecir_probabilidades(x)
            pred = np.argmax(probs)

            predicciones.append(pred)
            probabilidades_todas.append(probs)

            # Solo mostrar las primeras muestras si hay muchas
            if i < n_mostrar:
                estado = "✓ CORRECTO" if pred == y_real else "✗ INCORRECTO"

                print(f"Muestra {i+1}:")
                if nombres_caracteristicas:
                    for j, nombre in enumerate(nombres_caracteristicas):
                        print(f"  {nombre}: {x[j]:.2f}")
                else:
                    print(f"  Características: {x}")

                print(f"  Probabilidades softmax:")
                print(f"    P(clase 0 | x) = {probs[0]:.4f} ({probs[0]*100:.2f}%)")
                print(f"    P(clase 1 | x) = {probs[1]:.4f} ({probs[1]*100:.2f}%)")
                print(f"  Predicción: Clase {pred} ({'No Fraude' if pred == 0 else 'FRAUDE'})")
                print(f"  Real: Clase {y_real} ({'No Fraude' if y_real == 0 else 'FRAUDE'})")
                print(f"  {estado}\n")

        if len(X) > n_mostrar:
            print(f"... (mostrando solo {n_mostrar} de {len(X)} muestras)\n")

        aciertos = sum([1 for pred, real in zip(predicciones, y) if pred == real])
        precision = aciertos / len(y) * 100

        print(f"{'='*70}")
        print(f"RESULTADOS FINALES")
        print(f"{'='*70}")
        print(f"Aciertos: {aciertos}/{len(y)}")
        print(f"Precisión: {precision:.2f}%")
        print(f"{'='*70}\n")

        return predicciones, precision, probabilidades_todas


# ============================================
# FUNCIÓN: RECEPTOR DE ARCHIVOS CSV
# ============================================

def cargar_csv():
    """
    Función para cargar un archivo CSV desde Google Colab
    Retorna un DataFrame de pandas con los datos
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "CARGADOR DE ARCHIVOS CSV" + " "*25 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

    print("Por favor, selecciona tu archivo CSV...")
    print("El archivo debe contener:")
    print("  • Una columna con la etiqueta (0 o 1) llamada 'fraude' o similar")
    print("  • Las demás columnas serán las características\n")

    # Cargar archivo
    uploaded = files.upload()

    # Obtener el nombre del archivo
    filename = list(uploaded.keys())[0]

    print(f"\n✓ Archivo '{filename}' cargado exitosamente!\n")

    # Leer el CSV
    df = pd.read_csv(filename)

    return df, filename


# ============================================
# FUNCIÓN: ANALIZADOR DE CSV
# ============================================

def analizar_csv(df):
    """
    Analiza el contenido del CSV y extrae información relevante
    Retorna X (características), y (etiquetas), y nombres de características
    """
    print("\n" + "="*70)
    print("ANÁLISIS DEL ARCHIVO CSV")
    print("="*70 + "\n")

    # Mostrar información básica
    print(f"📊 Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas\n")

    print("📋 Columnas encontradas:")
    print("-" * 70)
    for i, col in enumerate(df.columns, 1):
        tipo_dato = df[col].dtype
        valores_unicos = df[col].nunique()
        print(f"{i}. {col:30s} | Tipo: {str(tipo_dato):10s} | Valores únicos: {valores_unicos}")
    print("-" * 70 + "\n")

    # Mostrar primeras filas
    print("🔍 Primeras 5 filas del dataset:")
    print(df.head())
    print("\n")

    # Identificar columna de etiquetas
    print("🎯 Identificando columna de etiquetas...")
    columnas_posibles = ['fraude', 'fraud', 'label', 'class', 'target', 'y', 'etiqueta']
    columna_etiqueta = None

    for col in df.columns:
        if col.lower() in columnas_posibles:
            columna_etiqueta = col
            break

    # Si no encuentra, buscar columna binaria
    if columna_etiqueta is None:
        for col in df.columns:
            if df[col].nunique() == 2:
                valores = sorted(df[col].unique())
                if valores == [0, 1]:
                    columna_etiqueta = col
                    print(f"⚠ No se encontró columna con nombre estándar.")
                    print(f"✓ Se detectó '{col}' como columna de etiquetas (valores: 0, 1)")
                    break

    if columna_etiqueta is None:
        print("\n❌ ERROR: No se pudo identificar la columna de etiquetas.")
        print("Por favor, asegúrate de que tu CSV tenga una columna llamada:")
        print("'fraude', 'fraud', 'label', 'class', 'target', o 'y'")
        print("O una columna binaria con valores 0 y 1\n")
        return None, None, None

    print(f"✓ Columna de etiquetas identificada: '{columna_etiqueta}'\n")

    # Extraer etiquetas
    y = df[columna_etiqueta].values

    # Verificar que las etiquetas sean 0 y 1
    valores_unicos = np.unique(y)
    if not np.array_equal(valores_unicos, [0, 1]):
        print(f"⚠ Advertencia: Las etiquetas no son exactamente [0, 1]")
        print(f"  Valores encontrados: {valores_unicos}")
        print(f"  Convirtiendo a 0 y 1...")
        # Convertir a 0 y 1
        y = (y != valores_unicos[0]).astype(int)

    # Extraer características (todas las columnas excepto la de etiquetas)
    columnas_caracteristicas = [col for col in df.columns if col != columna_etiqueta]
    X = df[columnas_caracteristicas].values

    print(f"📈 Estadísticas de las etiquetas:")
    print(f"  • Clase 0 (No Fraude): {sum(y == 0)} muestras ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  • Clase 1 (Fraude): {sum(y == 1)} muestras ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  • Total: {len(y)} muestras\n")

    print(f"🎨 Características para entrenar ({len(columnas_caracteristicas)}):")
    for i, col in enumerate(columnas_caracteristicas, 1):
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        print(f"  {i}. {col:30s} | Rango: [{min_val:.2f}, {max_val:.2f}] | Media: {mean_val:.2f}")

    print(f"\n{'='*70}")
    print("✓ ANÁLISIS COMPLETADO")
    print("="*70 + "\n")

    return X, y, columnas_caracteristicas


# ============================================
# PROGRAMA PRINCIPAL
# ============================================

print("\n" + "█"*70)
print("█" + " "*68 + "█")
print("█" + " "*15 + "PERCEPTRÓN - DETECCIÓN DE FRAUDES" + " "*20 + "█")
print("█" + " "*20 + "CON FUNCIÓN SOFTMAX" + " "*29 + "█")
print("█" + " "*68 + "█")
print("█"*70 + "\n")

print("Este programa implementa un perceptrón con función de activación Softmax")
print("para clasificación binaria de transacciones fraudulentas.\n")

# Paso 1: Cargar el archivo CSV
df, filename = cargar_csv()

# Paso 2: Analizar el CSV
X_fraude, y_fraude, nombres_caracteristicas = analizar_csv(df)

if X_fraude is None:
    print("❌ Error al procesar el archivo. Verifica el formato del CSV.")
else:
    # Paso 3: Configurar parámetros de entrenamiento
    print("\n" + "="*70)
    print("CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("="*70 + "\n")

    # Puedes ajustar estos valores
    TASA_APRENDIZAJE = 0.01
    ITERACIONES = 150  # Más de 10 como se requiere

    print(f"Parámetros seleccionados:")
    print(f"  • Tasa de aprendizaje: {TASA_APRENDIZAJE}")
    print(f"  • Número de iteraciones: {ITERACIONES}")
    print(f"  • Algoritmo: Descenso de gradiente con Softmax")
    print(f"  • Función de pérdida: Entropía cruzada\n")

    # Paso 4: Crear y entrenar el perceptrón
    perceptron_fraude = PerceptronSoftmax(
        tasa_aprendizaje=TASA_APRENDIZAJE,
        iteraciones=ITERACIONES
    )
    perceptron_fraude.entrenar(X_fraude, y_fraude)

    # Paso 5: Evaluar el perceptrón
    predicciones, precision, probabilidades = perceptron_fraude.evaluar(
        X_fraude, y_fraude, nombres_caracteristicas, mostrar_todas=False
    )

    # ============================================
    # GRÁFICO 1: Evolución del Error y Pérdida
    # ============================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subgráfico 1: Errores
    ax1.plot(range(1, perceptron_fraude.iteraciones + 1),
             perceptron_fraude.errores_por_epoca,
             marker='o',
             linestyle='-',
             color='red',
             linewidth=2,
             markersize=4)
    ax1.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Número de Errores', fontsize=12, fontweight='bold')
    ax1.set_title('Evolución de Errores durante el Entrenamiento',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Sin errores')
    ax1.legend(fontsize=10)

    # Subgráfico 2: Pérdida
    ax2.plot(range(1, perceptron_fraude.iteraciones + 1),
             perceptron_fraude.perdida_por_epoca,
             marker='s',
             linestyle='-',
             color='blue',
             linewidth=2,
             markersize=4)
    ax2.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pérdida (Cross-Entropy)', fontsize=12, fontweight='bold')
    ax2.set_title('Evolución de la Pérdida durante el Entrenamiento',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Pérdida'], fontsize=10)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("EXPLICACIÓN DEL GRÁFICO 1: EVOLUCIÓN DEL ERROR Y PÉRDIDA")
    print("="*70)
    print("""
Este gráfico dual muestra dos métricas importantes del entrenamiento:

GRÁFICO IZQUIERDO - ERRORES:
¿Qué está pasando?
- Muestra cuántas transacciones se clasificaron incorrectamente en cada época
- Al inicio hay muchos errores porque los pesos son aleatorios
- A medida que aprende, los errores disminuyen progresivamente
- Cuando llega cerca de 0, el modelo ha aprendido a clasificar correctamente
- Si no llega a 0, puede ser que los datos no sean perfectamente separables

GRÁFICO DERECHO - PÉRDIDA (CROSS-ENTROPY):
¿Qué está pasando?
- La pérdida mide qué tan "confiado" está el modelo en sus predicciones
- Usa la función de entropía cruzada: -log(p) donde p es la probabilidad correcta
- Una pérdida baja significa que el modelo da probabilidades altas a la clase correcta
- Incluso cuando los errores son 0, la pérdida puede seguir bajando
- Esto indica que el modelo está cada vez más "seguro" de sus predicciones

Importancia de Softmax:
- Softmax convierte las salidas en probabilidades reales (suman 1)
- Permite medir la confianza del modelo, no solo si acertó o no
- Para fraudes, es crucial saber QUÉ TAN seguro está el modelo
- Una probabilidad de 51% vs 99% de fraude hace gran diferencia en la práctica
    """)
    print("="*70 + "\n")

    # ============================================
    # GRÁFICO 2: Distribución de Probabilidades
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extraer probabilidades
    probs_clase_0 = [p[0] for p in probabilidades]
    probs_clase_1 = [p[1] for p in probabilidades]

    # Separar por clase real
    probs_legitimas_clase0 = [probs_clase_0[i] for i in range(len(y_fraude)) if y_fraude[i] == 0]
    probs_legitimas_clase1 = [probs_clase_1[i] for i in range(len(y_fraude)) if y_fraude[i] == 0]
    probs_fraudulentas_clase0 = [probs_clase_0[i] for i in range(len(y_fraude)) if y_fraude[i] == 1]
    probs_fraudulentas_clase1 = [probs_clase_1[i] for i in range(len(y_fraude)) if y_fraude[i] == 1]

    # Gráfico 1: Histograma de probabilidades para transacciones legítimas
    axes[0, 0].hist([probs_legitimas_clase0, probs_legitimas_clase1],
                    bins=20, label=['P(No Fraude)', 'P(Fraude)'],
                    color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Probabilidad', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Distribución de Probabilidades\n(Transacciones Legítimas)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Gráfico 2: Histograma de probabilidades para transacciones fraudulentas
    axes[0, 1].hist([probs_fraudulentas_clase0, probs_fraudulentas_clase1],
                    bins=20, label=['P(No Fraude)', 'P(Fraude)'],
                    color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Probabilidad', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Distribución de Probabilidades\n(Transacciones Fraudulentas)',
                         fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Gráfico 3: Scatter plot de confianza
    indices = np.arange(len(y_fraude))
    axes[1, 0].scatter(indices[y_fraude == 0],
                       [probs_clase_1[i] for i in indices[y_fraude == 0]],
                       c='green', s=50, alpha=0.6, label='Legítimas', edgecolors='black')
    axes[1, 0].scatter(indices[y_fraude == 1],
                       [probs_clase_1[i] for i in indices[y_fraude == 1]],
                       c='red', s=50, alpha=0.6, label='Fraudulentas', edgecolors='black')
    axes[1, 0].axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label='Umbral (0.5)')
    axes[1, 0].set_xlabel('Índice de Muestra', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('P(Fraude)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Confianza del Modelo por Muestra', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Gráfico 4: Matriz de confusión
    matriz_confusion = np.zeros((2, 2))
    for real, pred in zip(y_fraude, predicciones):
        matriz_confusion[real][pred] += 1

    im = axes[1, 1].imshow(matriz_confusion, cmap='Blues', aspect='auto')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['No Fraude', 'Fraude'], fontsize=11)
    axes[1, 1].set_yticklabels(['No Fraude', 'Fraude'], fontsize=11)
    axes[1, 1].set_xlabel('Predicción', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Valor Real', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Matriz de Confusión', fontsize=12, fontweight='bold')

    # Añadir valores en las celdas
    for i in range(2):
        for j in range(2):
            text = axes[1, 1].text(j, i, int(matriz_confusion[i, j]),
                                  ha="center", va="center", color="black",
                                  fontsize=20, fontweight='bold')

    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("EXPLICACIÓN DEL GRÁFICO 2: DISTRIBUCIÓN DE PROBABILIDADES")
    print("="*70)
    print("""
Este conjunto de 4 gráficos analiza el comportamiento probabilístico del modelo:

GRÁFICO SUPERIOR IZQUIERDO - Transacciones Legítimas:
- Muestra las probabilidades asignadas a transacciones realmente legítimas
- Verde: P(No Fraude) - debe ser ALTA (cercana a 1)
- Roja: P(Fraude) - debe ser BAJA (cercana a 0)
- Si el modelo funciona bien, el verde domina y está cerca de 1

GRÁFICO SUPERIOR DERECHO - Transacciones Fraudulentas:
- Muestra las probabilidades asignadas a transacciones realmente fraudulentas
- Verde: P(No Fraude) - debe ser BAJA (cercana a 0)
- Roja: P(Fraude) - debe ser ALTA (cercana a 1)
- Si el modelo funciona bien, el rojo domina y está cerca de 1

GRÁFICO INFERIOR IZQUIERDO - Confianza por Muestra:
- Cada punto es una transacción
- Eje Y: Probabilidad de ser fraude
- Puntos verdes (legítimas) deben estar abajo del umbral 0.5
- Puntos rojos (fraudes) deben estar arriba del umbral 0.5
- Muestra qué tan "seguro" está el modelo de cada predicción

GRÁFICO INFERIOR DERECHO - Matriz de Confusión:
- Fila = Clase real, Columna = Predicción
- Diagonal principal (azul oscuro) = Predicciones correctas
- Fuera de diagonal = Errores
- Ideal: números grandes en diagonal, ceros fuera de ella
    • [0,0]: Legítimas correctamente clasificadas (Verdaderos Negativos)
    • [1,1]: Fraudes correctamente detectados (Verdaderos Positivos)
    • [0,1]: Legítimas marcadas como fraude (Falsos Positivos)
    • [1,0]: Fraudes no detectados (Falsos Negativos) - MUY PELIGROSO
    """)
    print("="*70 + "\n")

    # ============================================
    # GRÁFICO 3: Importancia de Características
    # ============================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Calcular importancia como diferencia absoluta de pesos entre clases
    importancia = np.abs(perceptron_fraude.pesos[:, 1] - perceptron_fraude.pesos[:, 0])

    # Crear colores para las barras
    colores_barras = plt.cm.viridis(np.linspace(0.3, 0.9, len(nombres_caracteristicas)))