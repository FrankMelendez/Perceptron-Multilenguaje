#include <bits/stdc++.h>
using namespace std;

// Función de activación ReLU
double relu(double x) {
    return (x > 0) ? x : 0;
}

// Derivada de ReLU
double relu_deriv(double x) {
    return (x > 0) ? 1 : 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ====== 1. Leer CSV ======
    ifstream file("seattle-weather.csv");
    if (!file.is_open()) {
        cerr << "No se pudo abrir el archivo CSV.\n";
        return 1;
    }

    string line;
    vector<vector<double>> X; 
    vector<int> y;

    getline(file, line); // saltar cabecera
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<double> row;
        int col = 0;
        int label = 0;

        while (getline(ss, token, ',')) {
            if (col == 1 || col == 2 || col == 3 || col == 4) {
                // columnas numéricas: precipitation, temp_max, temp_min, wind
                row.push_back(stod(token));
            }
            if (col == 5) {
                // columna weather (categoría)
                // simplificamos: "rain"=1, otros=0
                label = (token.find("rain") != string::npos);
            }
            col++;
        }
        if (!row.empty()) {
            X.push_back(row);
            y.push_back(label);
        }
    }
    file.close();

    int n_samples = X.size();
    int n_features = X[0].size();

    // ====== 2. Inicializar pesos ======
    vector<double> W(n_features);
    double b = 0.0;
    srand(time(0));
    for (int i = 0; i < n_features; i++) {
        W[i] = ((double)rand() / RAND_MAX) - 0.5; // aleatorio -0.5 a 0.5
    }

    double lr = 0.001;   // tasa de aprendizaje
    int epochs = 20;     // más de 10 iteraciones
    vector<double> errores;

    // ====== 3. Entrenamiento ======
    for (int e = 0; e < epochs; e++) {
        double total_error = 0;
        for (int i = 0; i < n_samples; i++) {
            double z = b;
            for (int j = 0; j < n_features; j++) {
                z += X[i][j] * W[j];
            }
            double pred = relu(z);

            // binarizar salida: 1 si >0.5, 0 si no
            int pred_label = (pred > 0.5) ? 1 : 0;
            double error = y[i] - pred;

            total_error += fabs(error);

            // Gradientes
            double grad = error * relu_deriv(z);
            for (int j = 0; j < n_features; j++) {
                W[j] += lr * grad * X[i][j];
            }
            b += lr * grad;
        }
        errores.push_back(total_error / n_samples);
        cout << "Epoch " << e+1 << " - Error medio: " << errores.back() << "\n";
    }

    // ====== 4. Guardar errores para graficar ======
    ofstream fout("errores.dat");
    for (int i = 0; i < errores.size(); i++) {
        fout << i+1 << " " << errores[i] << "\n";
    }
    fout.close();

    return 0;
}