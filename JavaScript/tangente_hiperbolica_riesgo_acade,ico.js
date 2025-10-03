%%writefile perceptron.js
// ============================================
// Perceptrón en JS con activación tanh
// ============================================
const fs = require('fs');

// Cargar dataset
let raw = fs.readFileSync("data.json");
let data = JSON.parse(raw);
let X = data.X;
let y = data.y;

// Normalizar etiquetas a -1 y 1 (para tanh)
y = y.map(v => v === 0 ? -1 : 1);

// Hiperparámetros
const lr = 0.001;
const epochs = 20;

// Inicialización de pesos
let weights = new Array(X[0].length).fill(0).map(() => Math.random() - 0.5);
let bias = Math.random() - 0.5;

// Función de activación tanh
function activation(z) {
  return Math.tanh(z);
}

// Guardar errores por época
let errors = [];

// Entrenamiento
for (let epoch = 1; epoch <= epochs; epoch++) {
  let totalError = 0;
  for (let i = 0; i < X.length; i++) {
    let z = X[i].reduce((acc, val, j) => acc + val * weights[j], bias);
    let output = activation(z);
    let error = y[i] - output;

    // Actualización de pesos
    for (let j = 0; j < weights.length; j++) {
      weights[j] += lr * error * X[i][j];
    }
    bias += lr * error;
    totalError += Math.abs(error);
  }
  errors.push(totalError);
  console.log(Epoch ${epoch}, Error: ${totalError.toFixed(4)});
}

// Predicciones finales
let predictions = [];
for (let i = 0; i < X.length; i++) {
  let z = X[i].reduce((acc, val, j) => acc + val * weights[j], bias);
  let pred = activation(z);
  predictions.push({input: X[i], real: y[i], pred: pred});
}

// Guardar resultados en JSON
fs.writeFileSync("results.json", JSON.stringify({
  errors: errors,
  predictions: predictions,
  weights: weights,
  bias: bias
}, null, 2));

console.log("\n✅ Resultados guardados en results.json");
