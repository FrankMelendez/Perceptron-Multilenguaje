%%writefile perceptron.js
const fs = require("fs");
const parse = require("csv-parse/sync");

// Función de activación: tangente hiperbólica
function activation(x) {
    return Math.tanh(x);
}

// Derivada de tanh
function activationDerivative(x) {
    return 1 - Math.pow(Math.tanh(x), 2);
}

// Clase Perceptrón
class Perceptron {
    constructor(inputSize, learningRate = 0.1) {
        this.weights = Array(inputSize).fill(0).map(() => Math.random() - 0.5);
        this.bias = Math.random() - 0.5;
        this.learningRate = learningRate;
    }

    predict(inputs) {
        const sum = this.weights.reduce((acc, w, i) => acc + w * inputs[i], this.bias);
        return activation(sum);
    }

    train(trainingData, epochs = 50) {
        let errors = [];
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            trainingData.forEach(([inputs, target]) => {
                const output = this.predict(inputs);
                const error = target - output;
                totalError += error ** 2;

                // Actualización de pesos
                const gradient = error * activationDerivative(output);
                this.weights = this.weights.map((w, i) => w + this.learningRate * gradient * inputs[i]);
                this.bias += this.learningRate * gradient;
            });
            errors.push(totalError);
            console.log(Epoch ${epoch+1}, Error: ${totalError.toFixed(4)});
        }
        return errors;
    }
}

// --- CARGA DEL CSV ---
const file = fs.readFileSync("dataset.csv", "utf8");
const records = parse.parse(file, {columns: false, skip_empty_lines: true});

// Suponiendo que la última columna es la salida
const data = records.map(row => {
    const inputs = row.slice(0, row.length-1).map(Number);
    const target = Number(row[row.length-1]);
    return [inputs, target];
});

// Inicializar perceptrón
const inputSize = data[0][0].length;
const perceptron = new Perceptron(inputSize, 0.1);

// Entrenar
const errors = perceptron.train(data, 50);

// Guardar errores en CSV
fs.writeFileSync("errors.csv", errors.map((e,i) => ${i+1},${e}).join("\n"));

// Guardar función de activación
let xs = Array.from({length: 100}, (_, i) => -5 + i*0.1);
let ys = xs.map(activation);
fs.writeFileSync("activation.csv", xs.map((x,i) => ${x},${ys[i]}).join("\n"));

console.log("Resultados guardados en errors.csv y activation.csv");