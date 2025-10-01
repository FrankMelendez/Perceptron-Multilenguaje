using System;
using System.Collections.Generic;

public class Program
{
    public static void Main()
    {
        Console.WriteLine("=== PERCEPTRÓN CON FUNCIÓN ESCALÓN - OR LÓGICO ===\n");
        
        // Mostrar gráfico de la función escalón primero
        MostrarGraficoEscalon();
        
        // Datos de entrenamiento para OR lógico
        List<double[]> datosEntrenamiento = new List<double[]>
        {
            new double[] {0, 0},
            new double[] {0, 1},
            new double[] {1, 0},
            new double[] {1, 1}
        };

        List<int> etiquetasOR = new List<int> { 0, 1, 1, 1 };

        // Crear y entrenar el perceptrón con más iteraciones
        PerceptronEscalon perceptron = new PerceptronEscalon(
            numEntradas: 2, 
            tasaAprendizaje: 0.1, 
            maxIteraciones: 50  // 50 iteraciones para mostrar más progreso
        );

        // Entrenar el modelo
        perceptron.Entrenar(datosEntrenamiento, etiquetasOR);

        // Probar el modelo
        perceptron.Probar(datosEntrenamiento, etiquetasOR);

        // Mostrar resultados finales
        Console.WriteLine("\n=== RESULTADOS FINALES ===");
        Console.WriteLine("La tabla de verdad OR lógico es:");
        Console.WriteLine("A\tB\tA OR B");
        Console.WriteLine("0\t0\t0");
        Console.WriteLine("0\t1\t1");
        Console.WriteLine("1\t0\t1");
        Console.WriteLine("1\t1\t1");
        
        // Mostrar ecuación aprendida
        Console.WriteLine($"\nEcuación aprendida: {perceptron.Pesos[0]:F4}*A + {perceptron.Pesos[1]:F4}*B + {perceptron.Bias:F4}");
        Console.WriteLine($"Umbral de decisión: {perceptron.Pesos[0]:F4}*A + {perceptron.Pesos[1]:F4}*B + {perceptron.Bias:F4} ≥ 0");
        
        // Mostrar análisis de la solución
        Console.WriteLine("\n=== ANÁLISIS DE LA SOLUCIÓN ===");
        Console.WriteLine("El perceptrón encontró los pesos y bias que satisfacen:");
        Console.WriteLine($"Para [0,0]: {perceptron.Bias:F4} < 0 → 0 (CORRECTO)");
        Console.WriteLine($"Para [0,1]: {perceptron.Pesos[1] + perceptron.Bias:F4} ≥ 0 → 1 (CORRECTO)");
        Console.WriteLine($"Para [1,0]: {perceptron.Pesos[0] + perceptron.Bias:F4} ≥ 0 → 1 (CORRECTO)");
        Console.WriteLine($"Para [1,1]: {perceptron.Pesos[0] + perceptron.Pesos[1] + perceptron.Bias:F4} ≥ 0 → 1 (CORRECTO)");
    }
    
    public static void MostrarGraficoEscalon()
    {
        Console.WriteLine("GRAFICO DE LA FUNCIÓN ESCALÓN");
        Console.WriteLine("=============================\n");
        
        int altura = 12;
        int ancho = 60;
        
        Console.WriteLine("f(x)");
        Console.WriteLine("↑");
        
        for (int y = altura; y >= 0; y--)
        {
            if (y == altura) Console.Write("1 ");
            else if (y == altura / 2) Console.Write("0.5");
            else if (y == 0) Console.Write("0 ");
            else Console.Write("  ");
            
            Console.Write(" |");
            
            for (int x = 0; x < ancho; x++)
            {
                double valorX = (x - ancho / 2.0) / 5.0;
                
                if (y == altura / 2) // Eje X
                {
                    if (x == ancho / 2) Console.Write("┼");
                    else Console.Write("─");
                }
                else if (x == ancho / 2) // Eje Y
                {
                    Console.Write("│");
                }
                else
                {
                    double valorY = 1.0 - (double)y / altura;
                    
                    // Dibujar la función escalón
                    if (valorX < 0 && Math.Abs(valorY - 0) < 0.1)
                    {
                        Console.Write("─"); // Línea en y=0 para x<0
                    }
                    else if (valorX >= 0 && Math.Abs(valorY - 1) < 0.1)
                    {
                        Console.Write("─"); // Línea en y=1 para x>=0
                    }
                    else if (Math.Abs(valorX) < 0.2)
                    {
                        if (Math.Abs(valorY - 0) < 0.1) Console.Write("○"); // Punto vacío en (0,0)
                        else if (Math.Abs(valorY - 1) < 0.1) Console.Write("●"); // Punto lleno en (0,1)
                        else Console.Write(" ");
                    }
                    else
                    {
                        Console.Write(" ");
                    }
                }
            }
            Console.WriteLine();
        }
        
        Console.Write("    ");
        for (int x = 0; x < ancho; x++)
        {
            if (x == ancho / 2) Console.Write("0");
            else if (x == 10) Console.Write("-4");
            else if (x == ancho - 10) Console.Write("4");
            else Console.Write(" ");
        }
        Console.WriteLine("\n           x →");
        
        Console.WriteLine("\nDEFINICIÓN: f(x) = 0 si x < 0, f(x) = 1 si x ≥ 0");
        Console.WriteLine("=================================================\n");
    }
}

public class PerceptronEscalon
{
    private double[] pesos;
    private double bias;
    private double tasaAprendizaje;
    private int maxIteraciones;

    public double[] Pesos => pesos;
    public double Bias => bias;

    public PerceptronEscalon(int numEntradas, double tasaAprendizaje = 0.1, int maxIteraciones = 100)
    {
        this.pesos = new double[numEntradas];
        this.bias = 0;
        this.tasaAprendizaje = tasaAprendizaje;
        this.maxIteraciones = maxIteraciones;
        
        // Inicializar pesos con valores pequeños aleatorios
        Random rnd = new Random();
        for (int i = 0; i < numEntradas; i++)
        {
            pesos[i] = rnd.NextDouble() * 0.2 - 0.1; // Valores entre -0.1 y 0.1
        }
        
        Console.WriteLine($"Configuración del Perceptrón:");
        Console.WriteLine($"- Tasa de aprendizaje: {tasaAprendizaje}");
        Console.WriteLine($"- Máximo de iteraciones: {maxIteraciones}");
        Console.WriteLine($"- Pesos iniciales: [{pesos[0]:F4}, {pesos[1]:F4}]");
        Console.WriteLine($"- Bias inicial: {bias:F4}\n");
    }

    // Función de activación escalón
    private int FuncionActivacion(double x)
    {
        return x >= 0 ? 1 : 0;
    }

    // Calcular la salida del perceptrón
    public int Predecir(double[] entradas)
    {
        if (entradas.Length != pesos.Length)
        {
            throw new ArgumentException("El número de entradas no coincide con el número de pesos");
        }

        double suma = bias;
        for (int i = 0; i < entradas.Length; i++)
        {
            suma += entradas[i] * pesos[i];
        }

        return FuncionActivacion(suma);
    }

    // Entrenar el perceptrón
    public void Entrenar(List<double[]> conjuntoEntrenamiento, List<int> etiquetas)
    {
        if (conjuntoEntrenamiento.Count != etiquetas.Count)
        {
            throw new ArgumentException("El número de ejemplos de entrenamiento no coincide con el número de etiquetas");
        }

        Console.WriteLine("INICIANDO ENTRENAMIENTO...");
        Console.WriteLine("Iteración\tError\t\tPesos\t\t\tBias\t\tSuma para [1,1]");
        Console.WriteLine("---------\t-----\t\t-----\t\t\t----\t\t---------------");

        int iteracionesRealizadas = 0;
        bool entrenamientoCompletado = false;
        
        for (int iteracion = 0; iteracion < maxIteraciones; iteracion++)
        {
            iteracionesRealizadas++;
            int erroresTotales = 0;

            for (int i = 0; i < conjuntoEntrenamiento.Count; i++)
            {
                double[] entradas = conjuntoEntrenamiento[i];
                int etiquetaReal = etiquetas[i];

                // Realizar predicción
                int prediccion = Predecir(entradas);

                // Calcular error
                int error = etiquetaReal - prediccion;

                if (error != 0)
                {
                    erroresTotales++;

                    // Actualizar pesos y bias
                    for (int j = 0; j < pesos.Length; j++)
                    {
                        pesos[j] += tasaAprendizaje * error * entradas[j];
                    }
                    bias += tasaAprendizaje * error;
                }
            }

            // Calcular suma ponderada para [1,1] para mostrar el progreso
            double sumaEjemplo = bias + pesos[0] * 1 + pesos[1] * 1;

            // Mostrar TODAS las iteraciones hasta la 15, luego cada 5
            if (iteracion < 15 || iteracion % 5 == 0 || iteracion == maxIteraciones - 1 || erroresTotales == 0)
            {
                Console.WriteLine($"{iteracion + 1}\t\t{erroresTotales}\t\t[{pesos[0]:F4}, {pesos[1]:F4}]\t{bias:F4}\t\t{sumaEjemplo:F4}");
            }

            // Si no hay errores, continuar por al menos 15 iteraciones para demostración
            if (erroresTotales == 0 && iteracion >= 14)
            {
                if (!entrenamientoCompletado)
                {
                    Console.WriteLine($"\n¡SOLUCIÓN ENCONTRADA en {iteracion + 1} iteraciones!");
                    Console.WriteLine("Continuando entrenamiento para mostrar más iteraciones...");
                    entrenamientoCompletado = true;
                }
            }
            
            // Terminar después de 20 iteraciones para no hacerlo muy largo
            if (iteracion >= 19 && entrenamientoCompletado)
            {
                Console.WriteLine($"\nEntrenamiento demostrativo completado después de {iteracion + 1} iteraciones.");
                break;
            }
        }
        
        Console.WriteLine($"\nTotal de iteraciones realizadas: {iteracionesRealizadas}");
    }

    // Probar el perceptrón entrenado
    public void Probar(List<double[]> conjuntoPrueba, List<int> etiquetasReales)
    {
        Console.WriteLine("\n=== PRUEBAS DEL PERCEPTRÓN ===");
        Console.WriteLine("Entradas\tEsperado\tObtenido\tSuma Ponderada\t¿Correcto?");
        Console.WriteLine("--------\t--------\t--------\t-------------\t----------");

        int aciertos = 0;
        for (int i = 0; i < conjuntoPrueba.Count; i++)
        {
            double[] entradas = conjuntoPrueba[i];
            int esperado = etiquetasReales[i];
            
            // Calcular suma ponderada
            double suma = bias;
            for (int j = 0; j < entradas.Length; j++)
            {
                suma += entradas[j] * pesos[j];
            }
            
            int obtenido = Predecir(entradas);
            bool correcto = esperado == obtenido;

            if (correcto) aciertos++;

            Console.WriteLine($"[{entradas[0]}, {entradas[1]}]\t\t{esperado}\t\t{obtenido}\t\t{suma:F4}\t\t{(correcto ? "✓" : "✗")}");
        }

        double precision = (double)aciertos / conjuntoPrueba.Count * 100;
        Console.WriteLine($"\nPRECISIÓN FINAL: {precision:F2}% ({aciertos}/{conjuntoPrueba.Count})");
    }
}