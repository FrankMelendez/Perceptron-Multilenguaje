import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, tasa_aprendizaje=0.1, iteraciones=50):
        """
        Inicializa el perceptrón

        Parámetros:
        - tasa_aprendizaje: velocidad de ajuste de los pesos
        - iteraciones: número de épocas de entrenamiento (>10)
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None
        self.errores_por_epoca = []
        self.pesos_historia = []
        self.sesgo_historia = []

    def funcion_activacion(self, z):
        """Función de activación lineal"""
        return z

    def inicializar_pesos(self, n_caracteristicas):
        """Inicializa los pesos y el sesgo aleatoriamente"""
        self.pesos = np.random.randn(n_caracteristicas) * 0.01
        self.sesgo = 0.0

    def suma_ponderada(self, x):
        """Calcula la suma ponderada: w·x + b"""
        return np.dot(x, self.pesos) + self.sesgo

    def predecir(self, x):
        """Realiza una predicción para una entrada dada"""
        z = self.suma_ponderada(x)
        salida = self.funcion_activacion(z)
        # Para clasificación binaria con función lineal, usar umbral
        return 1 if salida >= 0 else 0

    def entrenar(self, X, y):
        """
        Entrena el perceptrón con los datos de entrada

        Parámetros:
        - X: matriz de características (n_muestras, n_caracteristicas)
        - y: vector de etiquetas (n_muestras,)
        """
        n_muestras, n_caracteristicas = X.shape
        self.inicializar_pesos(n_caracteristicas)

        print(f"{'='*60}")
        print(f"INICIANDO ENTRENAMIENTO DEL PERCEPTRÓN")
        print(f"{'='*60}")
        print(f"Función de activación: LINEAL (f(z) = z)")
        print(f"Número de iteraciones: {self.iteraciones}")
        print(f"Tasa de aprendizaje: {self.tasa_aprendizaje}")
        print(f"Pesos iniciales: {self.pesos}")
        print(f"Sesgo inicial: {self.sesgo:.4f}")
        print(f"{'='*60}\n")

        for epoca in range(self.iteraciones):
            errores = 0

            for i in range(n_muestras):
                x_i = X[i]
                y_i = y[i]

                # Suma ponderada
                z = self.suma_ponderada(x_i)

                # Función de activación lineal
                salida = self.funcion_activacion(z)

                # Calcular error
                error = y_i - salida

                # Actualizar pesos y sesgo
                self.pesos += self.tasa_aprendizaje * error * x_i
                self.sesgo += self.tasa_aprendizaje * error

                # Contar errores (basado en clasificación)
                prediccion = 1 if salida >= 0 else 0
                if prediccion != y_i:
                    errores += 1

            self.errores_por_epoca.append(errores)
            self.pesos_historia.append(self.pesos.copy())
            self.sesgo_historia.append(self.sesgo)

            if epoca < 5 or epoca % 10 == 0 or epoca == self.iteraciones - 1:
                print(f"Época {epoca + 1:3d}/{self.iteraciones} | Errores: {errores} | "
                      f"Pesos: [{self.pesos[0]:7.4f}, {self.pesos[1]:7.4f}] | "
                      f"Sesgo: {self.sesgo:7.4f}")

        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"Pesos finales: [{self.pesos[0]:.6f}, {self.pesos[1]:.6f}]")
        print(f"Sesgo final: {self.sesgo:.6f}")
        print(f"Errores en última época: {self.errores_por_epoca[-1]}")
        print(f"{'='*60}\n")

    def evaluar(self, X, y):
        """Evalúa el perceptrón en un conjunto de datos"""
        print(f"{'='*60}")
        print(f"EVALUACIÓN DEL MODELO")
        print(f"{'='*60}\n")

        predicciones = []
        detalles = []

        for i, (x, y_real) in enumerate(zip(X, y)):
            z = self.suma_ponderada(x)
            salida_lineal = self.funcion_activacion(z)
            pred = self.predecir(x)

            predicciones.append(pred)
            detalles.append({
                'entrada': x,
                'z': z,
                'salida_lineal': salida_lineal,
                'prediccion': pred,
                'real': y_real,
                'correcto': pred == y_real
            })

            estado = "✓ CORRECTO" if pred == y_real else "✗ INCORRECTO"
            print(f"Muestra {i+1}:")
            print(f"  Entrada: [{x[0]}, {x[1]}]")
            print(f"  z = w·x + b = ({self.pesos[0]:.4f})*{x[0]} + ({self.pesos[1]:.4f})*{x[1]} + {self.sesgo:.4f} = {z:.4f}")
            print(f"  f(z) = {salida_lineal:.4f} (función lineal)")
            print(f"  Clasificación: {pred} (umbral en 0)")
            print(f"  Valor real: {y_real}")
            print(f"  {estado}\n")

        aciertos = sum([1 for d in detalles if d['correcto']])
        precision = aciertos / len(y) * 100

        print(f"{'='*60}")
        print(f"RESULTADOS FINALES")
        print(f"{'='*60}")
        print(f"Aciertos: {aciertos}/{len(y)}")
        print(f"Precisión: {precision:.2f}%")
        print(f"{'='*60}\n")

        return predicciones, precision, detalles


# ============================================
# CASO DE PRUEBA 1: AND LÓGICO
# ============================================

print("\n" + "█"*70)
print("█" + " "*68 + "█")
print("█" + " "*20 + "PERCEPTRÓN - AND LÓGICO" + " "*25 + "█")
print("█" + " "*68 + "█")
print("█"*70 + "\n")

# Datos de entrenamiento para AND (con más muestras para mayor robustez)
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1, 0, 0, 0, 1])

print("TABLA DE VERDAD - AND LÓGICO:")
print("-" * 30)
print("| Entrada 1 | Entrada 2 | Salida |")
print("-" * 30)
print("|     0     |     0     |   0    |")
print("|     0     |     1     |   0    |")
print("|     1     |     0     |   0    |")
print("|     1     |     1     |   1    |")
print("-" * 30)
print("\nObjetivo: La salida es 1 SOLO cuando ambas entradas son 1\n")

# Crear y entrenar el perceptrón
perceptron_and = Perceptron(tasa_aprendizaje=0.1, iteraciones=50)
perceptron_and.entrenar(X_and, y_and)

# Evaluar el perceptrón
predicciones, precision, detalles = perceptron_and.evaluar(X_and, y_and)

# ============================================
# GRÁFICO 1: Evolución del Error
# ============================================
plt.figure(figsize=(12, 6))
plt.plot(range(1, perceptron_and.iteraciones + 1),
         perceptron_and.errores_por_epoca,
         marker='o',
         linestyle='-',
         color='blue',
         linewidth=2,
         markersize=6)
plt.xlabel('Época', fontsize=12, fontweight='bold')
plt.ylabel('Número de Errores', fontsize=12, fontweight='bold')
plt.title('Evolución del Error durante el Entrenamiento (AND Lógico)\nFunción de Activación: Lineal',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Sin errores')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("EXPLICACIÓN DEL GRÁFICO 1: EVOLUCIÓN DEL ERROR")
print("="*70)
print("""
Este gráfico muestra cómo disminuyen los errores de clasificación a lo largo
de las épocas de entrenamiento.

¿Qué está pasando?
- Eje X: Representa cada época (iteración completa sobre todos los datos)
- Eje Y: Número de errores de clasificación en esa época
- La línea azul muestra la evolución del aprendizaje del perceptrón

Interpretación:
- Al inicio, el perceptrón comete varios errores porque los pesos son aleatorios
- A medida que avanza el entrenamiento, los pesos se ajustan y los errores disminuyen
- Cuando la línea llega a 0, el perceptrón ha aprendido perfectamente la función AND
- Si la línea se mantiene en 0, significa que el modelo está convergiendo correctamente

En este caso, el perceptrón aprende la función AND porque es linealmente separable.
""")
print("="*70 + "\n")

# ============================================
# GRÁFICO 2: Frontera de Decisión
# ============================================
plt.figure(figsize=(10, 8))

# Puntos de datos
clase_0_x = []
clase_0_y = []
clase_1_x = []
clase_1_y = []

for i in range(4):  # Solo las 4 combinaciones únicas
    if y_and[i] == 0:
        clase_0_x.append(X_and[i][0])
        clase_0_y.append(X_and[i][1])
    else:
        clase_1_x.append(X_and[i][0])
        clase_1_y.append(X_and[i][1])

plt.scatter(clase_0_x, clase_0_y, c='red', marker='o', s=300,
            edgecolors='black', linewidth=2.5, label='Clase 0 (Salida: 0)', alpha=0.7)
plt.scatter(clase_1_x, clase_1_y, c='blue', marker='s', s=300,
            edgecolors='black', linewidth=2.5, label='Clase 1 (Salida: 1)', alpha=0.7)

# Frontera de decisión (línea donde w·x + b = 0)
if perceptron_and.pesos[1] != 0:
    x_linea = np.linspace(-0.5, 1.5, 100)
    y_linea = -(perceptron_and.pesos[0] * x_linea + perceptron_and.sesgo) / perceptron_and.pesos[1]
    plt.plot(x_linea, y_linea, 'g-', linewidth=3, label='Frontera de decisión')

    # Regiones de decisión con sombreado
    x_grid = np.linspace(-0.5, 1.5, 300)
    y_grid = np.linspace(-0.5, 1.5, 300)
    xx, yy = np.meshgrid(x_grid, y_grid)
    Z = perceptron_and.pesos[0] * xx + perceptron_and.pesos[1] * yy + perceptron_and.sesgo
    plt.contourf(xx, yy, Z, levels=[-1000, 0, 1000], colors=['mistyrose', 'lightblue'], alpha=0.3)

# Añadir etiquetas a los puntos
for i in range(4):
    plt.annotate(f'({X_and[i][0]},{X_and[i][1]})',
                xy=(X_and[i][0], X_and[i][1]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('Entrada 1 (x₁)', fontsize=12, fontweight='bold')
plt.ylabel('Entrada 2 (x₂)', fontsize=12, fontweight='bold')
plt.title('Clasificación AND Lógico - Función de Activación Lineal\nFrontera de Decisión',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("EXPLICACIÓN DEL GRÁFICO 2: FRONTERA DE DECISIÓN")
print("="*70)
print(f"""
Este gráfico visualiza cómo el perceptrón separa las dos clases en el espacio 2D.

¿Qué está pasando?
- Círculos rojos (○): Puntos donde la salida debe ser 0
- Cuadrados azules (■): Puntos donde la salida debe ser 1
- Línea verde: La frontera de decisión (donde w·x + b = 0)
- Región rosa: Zona donde el perceptrón predice 0
- Región azul: Zona donde el perceptrón predice 1

Ecuación de la frontera:
w₁·x₁ + w₂·x₂ + b = 0
{perceptron_and.pesos[0]:.4f}·x₁ + {perceptron_and.pesos[1]:.4f}·x₂ + {perceptron_and.sesgo:.4f} = 0

Interpretación:
- El perceptrón encuentra una línea recta que separa los puntos
- Cualquier punto a la derecha/arriba de la línea verde se clasifica como 1
- Cualquier punto a la izquierda/abajo de la línea verde se clasifica como 0
- La función AND es linealmente separable, por eso el perceptrón puede aprenderla

Nota importante:
- El punto (1,1) está solo en un lado (clase 1)
- Los otros tres puntos están en el otro lado (clase 0)
- Esta es la característica distintiva del AND lógico
""")
print("="*70 + "\n")

# ============================================
# GRÁFICO 3: Evolución de los Pesos
# ============================================
plt.figure(figsize=(12, 6))

pesos_historia_array = np.array(perceptron_and.pesos_historia)
plt.plot(range(1, perceptron_and.iteraciones + 1),
         pesos_historia_array[:, 0],
         marker='o',
         linestyle='-',
         color='red',
         linewidth=2,
         markersize=5,
         label='Peso w₁')
plt.plot(range(1, perceptron_and.iteraciones + 1),
         pesos_historia_array[:, 1],
         marker='s',
         linestyle='-',
         color='blue',
         linewidth=2,
         markersize=5,
         label='Peso w₂')
plt.plot(range(1, perceptron_and.iteraciones + 1),
         perceptron_and.sesgo_historia,
         marker='^',
         linestyle='-',
         color='green',
         linewidth=2,
         markersize=5,
         label='Sesgo (b)')

plt.xlabel('Época', fontsize=12, fontweight='bold')
plt.ylabel('Valor del Parámetro', fontsize=12, fontweight='bold')
plt.title('Evolución de Pesos y Sesgo durante el Entrenamiento',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("EXPLICACIÓN DEL GRÁFICO 3: EVOLUCIÓN DE PESOS Y SESGO")
print("="*70)
print(f"""
Este gráfico muestra cómo cambian los parámetros del perceptrón durante
el entrenamiento.

¿Qué está pasando?
- Línea roja (w₁): Peso asociado a la primera entrada
- Línea azul (w₂): Peso asociado a la segunda entrada
- Línea verde (b): Sesgo o bias del perceptrón

Valores finales:
- w₁ = {perceptron_and.pesos[0]:.6f}
- w₂ = {perceptron_and.pesos[1]:.6f}
- b = {perceptron_and.sesgo:.6f}

Interpretación:
- Los pesos comienzan con valores aleatorios pequeños (cerca de 0)
- En cada época, se ajustan según la regla de aprendizaje del perceptrón
- La regla de actualización es: w_nuevo = w_viejo + α·error·x
- Cuando los errores llegan a 0, los pesos se estabilizan (convergen)
- Los valores finales definen la frontera de decisión óptima

Significado de los pesos:
- Pesos positivos grandes: esa entrada tiene importancia para activar la neurona
- Si ambos w₁ y w₂ son positivos y similares, significa que ambas entradas
  deben estar activas (valor 1) para que la salida sea positiva
- El sesgo negativo requiere que la suma de entradas supere ese umbral
""")
print("="*70 + "\n")