import java.util.*;
import java.io.*;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;

public class SpamClassifierCompleteSigmoidal {

    // ==================== DATOS PARA GRÁFICOS ====================
    private static java.util.List<Double> errors = new java.util.ArrayList<>();
    private static java.util.List<Double> accuracies = new java.util.ArrayList<>();
    private static java.util.List<Double> sigmoidValues = new java.util.ArrayList<>();
    private static java.util.List<Double> zValues = new java.util.ArrayList<>();
    private static java.util.List<double[]> weightEvolution = new java.util.ArrayList<>();

    // ==================== CLASE PERCEPTRÓN SIGMOIDAL ====================
    static class PerceptronSigmoidal {
        private double[] weights;
        private double bias;
        private double learningRate;
        private int epochs;

        public PerceptronSigmoidal(int featureSize, double learningRate, int epochs) {
            this.weights = new double[featureSize];
            this.bias = 0.1;
            this.learningRate = learningRate;
            this.epochs = epochs;

            Random rand = new Random(42);
            for (int i = 0; i < featureSize; i++) {
                weights[i] = rand.nextDouble() * 0.02 - 0.01;
            }
        }

        // FUNCIÓN SIGMOIDAL
        public double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        public double sigmoidDerivative(double x) {
            double sig = sigmoid(x);
            return sig * (1 - sig);
        }

        public double predict(double[] features) {
            double sum = bias;
            for (int i = 0; i < features.length; i++) {
                sum += weights[i] * features[i];
            }
            return sigmoid(sum);
        }

        public void train(double[][] X, double[] y) {
            System.out.println("=== ENTRENAMIENTO PERCEPTRÓN SIGMOIDAL ===");
            System.out.println("Época\tError Promedio\tPrecisión");
            System.out.println("========================================");

            // Guardar pesos iniciales
            weightEvolution.add(weights.clone());

            for (int epoch = 1; epoch <= epochs; epoch++) {
                double totalError = 0;
                int correct = 0;

                for (int i = 0; i < X.length; i++) {
                    // FORWARD PASS
                    double prediction = predict(X[i]);
                    double error = y[i] - prediction;
                    totalError += Math.abs(error);

                    // Calcular precisión
                    int predictedClass = prediction > 0.5 ? 1 : 0;
                    if (predictedClass == (int)y[i]) {
                        correct++;
                    }

                    // BACKWARD PASS
                    double gradient = error * sigmoidDerivative(prediction);

                    // ACTUALIZAR PESOS
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * gradient * X[i][j];
                    }
                    bias += learningRate * gradient;
                }

                double avgError = totalError / X.length;
                double accuracy = (double) correct / X.length * 100;

                errors.add(avgError);
                accuracies.add(accuracy);
                weightEvolution.add(weights.clone());

                System.out.printf("%d\t%.4f\t\t%.2f%%%n", epoch, avgError, accuracy);
            }
        }

        // Generar datos para curva sigmoidal
        public void generateSigmoidData() {
            sigmoidValues.clear();
            zValues.clear();

            for (double z = -6.0; z <= 6.0; z += 0.12) {
                double sigmoid = sigmoid(z);
                sigmoidValues.add(sigmoid);
                zValues.add(z);
            }
        }
    }

    // ==================== CLASES PARA GRÁFICOS ====================

    // Gráfico 1: Función Sigmoidal
    static class SigmoidGraphPanel extends JPanel {
        public SigmoidGraphPanel() {
            setPreferredSize(new Dimension(500, 400));
            setBackground(Color.WHITE);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int width = getWidth();
            int height = getHeight();
            int padding = 60;
            int graphWidth = width - 2 * padding;
            int graphHeight = height - 2 * padding;

            // Título
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 16));
            g2.drawString("🎯 FUNCIÓN SIGMOIDAL σ(z) = 1/(1 + e^(-z))", width/2 - 180, 30);

            // Ejes
            g2.setColor(Color.GRAY);
            g2.drawLine(padding, height - padding, width - padding, height - padding);
            g2.drawLine(padding, height - padding, padding, padding);

            // Curva sigmoidal
            if (zValues.size() > 1) {
                g2.setColor(Color.BLUE);
                g2.setStroke(new BasicStroke(3));

                for (int i = 0; i < zValues.size() - 1; i++) {
                    double z1 = zValues.get(i);
                    double z2 = zValues.get(i + 1);
                    double sig1 = sigmoidValues.get(i);
                    double sig2 = sigmoidValues.get(i + 1);

                    int x1 = padding + (int)((z1 + 6) * graphWidth / 12);
                    int y1 = height - padding - (int)(sig1 * graphHeight);
                    int x2 = padding + (int)((z2 + 6) * graphWidth / 12);
                    int y2 = height - padding - (int)(sig2 * graphHeight);

                    g2.drawLine(x1, y1, x2, y2);
                }

                // Umbral de decisión
                g2.setColor(Color.RED);
                g2.setStroke(new BasicStroke(2, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{5}, 0));
                int decisionY = height - padding - (int)(0.5 * graphHeight);
                g2.drawLine(padding, decisionY, width - padding, decisionY);
                g2.drawString("Umbral: 0.5", width - 100, decisionY - 10);
            }
        }
    }

    // Gráfico 2: Evolución del Error y Precisión
    static class TrainingGraphPanel extends JPanel {
        public TrainingGraphPanel() {
            setPreferredSize(new Dimension(500, 400));
            setBackground(Color.WHITE);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int width = getWidth();
            int height = getHeight();
            int padding = 60;
            int graphWidth = width - 2 * padding;
            int graphHeight = height - 2 * padding;

            // Título
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 16));
            g2.drawString("📈 EVOLUCIÓN DEL ENTRENAMIENTO (15 ÉPOCAS)", width/2 - 180, 30);

            // Ejes
            g2.setColor(Color.GRAY);
            g2.drawLine(padding, height - padding, width - padding, height - padding);
            g2.drawLine(padding, height - padding, padding, padding);

            // Dibujar curvas de error y precisión
            if (errors.size() > 1 && accuracies.size() > 1) {
                // Curva de error (rojo)
                g2.setColor(Color.RED);
                g2.setStroke(new BasicStroke(2));
                drawCurve(g2, errors, padding, height - padding, graphWidth, graphHeight, 0.5, false);

                // Curva de precisión (verde)
                g2.setColor(Color.GREEN);
                g2.setStroke(new BasicStroke(2));
                drawCurve(g2, accuracies, padding, height - padding, graphWidth, graphHeight, 100, true);

                // Leyenda
                g2.setColor(Color.RED);
                g2.fillRect(width - 150, padding, 10, 10);
                g2.setColor(Color.BLACK);
                g2.drawString("Error", width - 130, padding + 10);

                g2.setColor(Color.GREEN);
                g2.fillRect(width - 150, padding + 20, 10, 10);
                g2.setColor(Color.BLACK);
                g2.drawString("Precisión (%)", width - 130, padding + 30);
            }
        }

        private void drawCurve(Graphics2D g2, java.util.List<Double> data, int paddingX, int paddingY,
                             int graphWidth, int graphHeight, double maxValue, boolean isPercentage) {
            for (int i = 0; i < data.size() - 1; i++) {
                double value1 = data.get(i);
                double value2 = data.get(i + 1);

                int x1 = paddingX + (i * graphWidth / (data.size() - 1));
                int y1 = paddingY - (int)((value1 / maxValue) * graphHeight);
                int x2 = paddingX + ((i + 1) * graphWidth / (data.size() - 1));
                int y2 = paddingY - (int)((value2 / maxValue) * graphHeight);

                g2.drawLine(x1, y1, x2, y2);
                g2.fillOval(x1 - 2, y1 - 2, 4, 4);
            }
        }
    }

    // Gráfico 3: Evolución de Pesos
    static class WeightsGraphPanel extends JPanel {
        public WeightsGraphPanel() {
            setPreferredSize(new Dimension(500, 400));
            setBackground(Color.WHITE);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int width = getWidth();
            int height = getHeight();
            int padding = 60;
            int graphWidth = width - 2 * padding;
            int graphHeight = height - 2 * padding;

            // Título
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Arial", Font.BOLD, 16));
            g2.drawString("⚖️ EVOLUCIÓN DE PESOS (w) Y BIAS (b)", width/2 - 150, 30);

            if (weightEvolution.size() > 1) {
                // Dibujar evolución de primeros 5 pesos
                Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.ORANGE, Color.MAGENTA};
                int maxWeightsToShow = Math.min(5, weightEvolution.get(0).length);

                for (int w = 0; w < maxWeightsToShow; w++) {
                    g2.setColor(colors[w % colors.length]);
                    g2.setStroke(new BasicStroke(2));

                    for (int epoch = 0; epoch < weightEvolution.size() - 1; epoch++) {
                        double weight1 = weightEvolution.get(epoch)[w];
                        double weight2 = weightEvolution.get(epoch + 1)[w];

                        int x1 = padding + (epoch * graphWidth / (weightEvolution.size() - 1));
                        int y1 = padding + (int)((1 - normalizeWeight(weight1)) * graphHeight);
                        int x2 = padding + ((epoch + 1) * graphWidth / (weightEvolution.size() - 1));
                        int y2 = padding + (int)((1 - normalizeWeight(weight2)) * graphHeight);

                        g2.drawLine(x1, y1, x2, y2);
                    }

                    // Leyenda
                    g2.fillRect(width - 100, padding + (w * 20), 10, 10);
                    g2.setColor(Color.BLACK);
                    g2.drawString("w" + (w + 1), width - 85, padding + 10 + (w * 20));
                }
            }
        }

        private double normalizeWeight(double weight) {
            // Normalizar peso entre 0 y 1 para visualización
            return (weight + 0.1) / 0.2;
        }
    }

    // ==================== MOSTRAR TODOS LOS GRÁFICOS ====================
    public static void showAllGraphics() {
        JFrame frame = new JFrame("🧮 PERCEPTRÓN SIGMOIDAL - Análisis Completo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(2, 2));

        // Gráfico 1: Función Sigmoidal
        SigmoidGraphPanel sigmoidGraph = new SigmoidGraphPanel();

        // Gráfico 2: Evolución del Entrenamiento
        TrainingGraphPanel trainingGraph = new TrainingGraphPanel();

        // Gráfico 3: Evolución de Pesos
        WeightsGraphPanel weightsGraph = new WeightsGraphPanel();

        // Panel de información
        JTextArea infoArea = new JTextArea(
            "🎯 PERCEPTRÓN SIGMOIDAL - 15 ITERACIONES\n\n" +
            "FÓRMULAS IMPLEMENTADAS:\n" +
            "• σ(z) = 1 / (1 + e^(-z))\n" +
            "• z = b + Σ(wᵢ * xᵢ)\n" +
            "• ∇ = error * σ'(z)\n" +
            "• wᵢ = wᵢ + η * ∇ * xᵢ\n\n" +
            "RESULTADOS ESPERADOS:\n" +
            "• Error decreciente\n" +
            "• Precisión creciente\n" +
            "• Pesos estabilizándose"
        );
        infoArea.setEditable(false);
        infoArea.setFont(new Font("Arial", Font.PLAIN, 12));
        infoArea.setBackground(Color.WHITE);

        frame.add(sigmoidGraph);
        frame.add(trainingGraph);
        frame.add(weightsGraph);
        frame.add(new JScrollPane(infoArea));

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    // ==================== MÉTODO PRINCIPAL ====================
    public static void main(String[] args) {
        System.out.println("🚀 PERCEPTRÓN SIGMOIDAL - 15 ITERACIONES COMPLETAS");
        System.out.println("=================================================\n");

        // Datos de entrenamiento ampliados
        String[] trainingData = {
            "ham,Subject: enron methanol meter follow up note monday",
            "spam,Subject: photoshop windows office cheap trending",
            "ham,Subject: hpl nom january see attached file",
            "spam,Subject: looking medication best source",
            "ham,Subject: indian springs deal book revenue",
            "spam,Subject: cheap pills medication viagra",
            "ham,Subject: meeting schedule next week conference",
            "spam,Subject: make money fast work home",
            "ham,Subject: project status report deliverables",
            "spam,Subject: lottery winner claim prize",
            "ham,Subject: report analysis data results",
            "spam,Subject: discount sale limited time",
            "ham,Subject: team meeting agenda topics",
            "spam,Subject: credit card offer apply now",
            "ham,Subject: client presentation materials ready"
        };

        try {
            // Cargar datos
            java.util.List<Email> emails = new java.util.ArrayList<>();
            for (String data : trainingData) {
                String[] parts = data.split(",", 2);
                if (parts.length == 2) {
                    int label = parts[0].equals("spam") ? 1 : 0;
                    emails.add(new Email(parts[1], label));
                }
            }

            // Procesador de texto simplificado
            TextProcessor processor = new TextProcessor();
            java.util.List<String> texts = new java.util.ArrayList<>();
            for (Email email : emails) texts.add(email.text);
            processor.buildVocabulary(texts);

            // Preparar datos
            double[][] X = new double[emails.size()][];
            double[] y = new double[emails.size()];
            for (int i = 0; i < emails.size(); i++) {
                X[i] = processor.textToFeatures(emails.get(i).text);
                y[i] = emails.get(i).label;
            }

            System.out.println("📊 DATOS DE ENTRENAMIENTO:");
            System.out.println("• Emails: " + emails.size());
            System.out.println("• Vocabulario: " + processor.getVocabSize() + " palabras");
            System.out.println("• Épocas: 15");
            System.out.println("• Learning Rate: 0.1\n");

            // Entrenar modelo por 15 épocas
            PerceptronSigmoidal model = new PerceptronSigmoidal(processor.getVocabSize(), 0.1, 15);
            model.generateSigmoidData(); // Generar datos para gráfico sigmoidal
            model.train(X, y);

            // Mostrar resultados finales
            System.out.println("\n=== 📈 RESULTADOS FINALES ===");
            System.out.printf("• Error final: %.4f%n", errors.get(errors.size()-1));
            System.out.printf("• Precisión final: %.2f%%%n", accuracies.get(accuracies.size()-1));
            System.out.println("• Total épocas: " + errors.size());

            // Mostrar gráficos
            javax.swing.SwingUtilities.invokeLater(() -> showAllGraphics());

        } catch (Exception e) {
            System.out.println("❌ Error: " + e.getMessage());
        }
    }

    // ==================== CLASES AUXILIARES ====================
    static class TextProcessor {
        private Map<String, Integer> vocabulary = new HashMap<>();
        private int vocabSize = 0;

        public void buildVocabulary(java.util.List<String> texts) {
            for (String text : texts) {
                String[] words = text.toLowerCase().split("\\s+");
                for (String word : words) {
                    if (!vocabulary.containsKey(word)) {
                        vocabulary.put(word, vocabSize++);
                    }
                }
            }
        }

        public double[] textToFeatures(String text) {
            double[] features = new double[vocabSize];
            String[] words = text.toLowerCase().split("\\s+");
            for (String word : words) {
                Integer index = vocabulary.get(word);
                if (index != null) features[index] = 1.0;
            }
            return features;
        }
    }

    static class Email {
        String text;
        int label;
        public Email(String text, int label) {
            this.text = text; this.label = label;
        }
    }
}