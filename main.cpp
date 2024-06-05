#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <xgboost/c_api.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

using namespace std;

// Безопасный вызов для функций из XGB C API
// Safe call for XGB C API functions
#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

// Метрика R2 (R squared metric)
float calculateR2(std::vector<float> x, std::vector<float> y) {
    float meanX = 0.0, meanY = 0.0;
    for (float value : x) {
        meanX += value;
    }
    meanX /= x.size();

    for (float value : y) {
        meanY += value;
    }
    meanY /= y.size();

    float SSx = 0.0, SSy = 0.0, Cov = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        SSx += std::pow((x[i] - meanX), 2);
        SSy += std::pow((y[i] - meanY), 2);
        Cov += (x[i] - meanX) * (y[i] - meanY);
    }

    float rSquared = std::pow(Cov, 2) / (SSx * SSy);
    return rSquared;
}

// Разбиение строки по запятым
// String comma split
std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }
    res.push_back(s.substr(pos_start));
    return res;
}


// Загрузка CSV-файла. Все колонки кроме последней - фичи.
// CSV file loading. Counting all columns are features except the last
void loadCSV(const std::string& filename, vector<int> test_indices, \
    vector<float> &train_features, vector<float> &test_features, vector<float> &train_target, vector<float>& test_target) {
    ifstream in(filename);     
    string line;
    vector<string> substrings; 
    
    int count = 0;
    bool is_test = false;
    if (in.fail()) {
        std::cout << "No file!" << endl;
    }
    if (in.is_open())
    {
        while (getline(in, line))
        {
           
            if (count > 0) {              
                substrings = split(line, ",");
                is_test = false;
                
                if (std::find(test_indices.begin(), test_indices.end(), (count -1)) != test_indices.end()) is_test = true;     
      

                int ii = 0;

                for (int i = 1; i < (substrings.size() - 1); ++i) {
                   // cout << substrings[i];
                    if (is_test) test_features.push_back(stof(substrings[i]));
                    else train_features.push_back(stof(substrings[i]));
                    ii = ii + 1;

                }
               // cout << endl;
                //std::cout << ii;
                if (is_test) test_target.push_back(stof(substrings.back()));
                else train_target.push_back(stof(substrings.back()));

            }
            count++;
           
        }
    }
    in.close();    
}

// Получение вектора размером how_much случайных целых от 0 до max_num
// Getting the random int (0..max_num) vector of size how_much 

vector<int> getRandomNumbers(int how_much, int max_num) {

    vector<int> numbers;
  

    for (int i = 0; i < max_num; ++i) {
        numbers.push_back(i);
    }

    random_device rd;  
    mt19937 g(rd());   

    std::shuffle(numbers.begin(), numbers.end(), g);

 
    vector<int> result (numbers.begin(), numbers.begin() + how_much);

    return result;
}

int main() {
    // Загрузка CSV файла
    // Loading CSV file
    std::vector<float> train_features;
    std::vector<float> train_target;
    std::vector<float> test_features;
    std::vector<float> test_target;


    // индексы для тестовой выборки
    // Test data indices
    std::vector<int> test_indices;
    const int MAX_ROWS = 506;
    auto test_size = int(MAX_ROWS / );
    test_indices = getRandomNumbers(test_size, MAX_ROWS);
        
    loadCSV("./Datasets/BostonHousing.csv", test_indices, train_features, test_features, train_target, test_target);


    std::cout << "Train features size: " << train_features.size() << ", Train target size: " << train_target.size() << endl;
    std::cout << "Test features size: " << test_features.size() << ", Test target size: " << test_target.size() << endl;
 
    // Создание матриц обучающей и тестовой выборки.
    // Train and test data matrices  
    int rows = static_cast<int> (train_target.size());
    int cols = int(train_features.size() / rows);        
    DMatrixHandle dtrain;
    safe_xgboost(XGDMatrixCreateFromMat(&train_features[0], rows, cols, -9999, &dtrain));
    safe_xgboost(XGDMatrixSetFloatInfo(dtrain, "label", &train_target[0], rows));
  
    std::cout << "Creation of matrix ok; " << "rows=" << rows << ";cols" << cols << endl;
    DMatrixHandle dtest;
    rows = static_cast<int> (test_target.size());
    cols = int(test_features.size() / rows);
    safe_xgboost(XGDMatrixCreateFromMat(&test_features[0], rows, cols, -9999, &dtest));
    safe_xgboost(XGDMatrixSetFloatInfo(dtest, "label", &test_target[0], rows));      
    std::cout << "Creation of matrix ok; " << "rows=" << rows << ";cols=" << cols << endl;

    // Совместная матрица 
    // Coupled matrix
    DMatrixHandle eval_dmats[2] = { dtrain, dtest };

    // Непосредственно модель
    // The model itself
    BoosterHandle booster;  

    // Параметры
    // Parameters
    safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));
    safe_xgboost(XGBoosterSetParam(booster, "booster", "gbtree"));  
    safe_xgboost(XGBoosterSetParam(booster, "device", "cpu"));
    safe_xgboost(XGBoosterSetParam(booster, "verbosity", "2"));
    safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1"));
    bst_ulong num_feature = 0;
    safe_xgboost(XGBoosterGetNumFeature(booster, &num_feature));
    std::printf("num_feature: %lu\n", (unsigned long)(num_feature));
 
    // Обучение модели из n_trees деревьев
    // Model training with n_trees trees
    int n_trees = 300;
    const char* eval_names[2] = { "train", "test" };
    const char* eval_result = NULL;
    for (int i = 0; i < n_trees; ++i) { 
        safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));      
    }
     
    // Предсказание на тестовой выборке
    // Prediction on test data
    bst_ulong out_len = 0;
    int n_print = 10;

    char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";    
    uint64_t const* out_shape;
    uint64_t out_dim;
    float const* out_result = NULL;
    
    safe_xgboost(
        XGBoosterPredictFromDMatrix(booster, dtest, config, &out_shape, &out_dim, &out_result));

    vector<float> pred_vector(out_result, out_result + test_size);

    // Вычисление R2
    // R2 calculation
    float r2metric = calculateR2(test_target, pred_vector);
    cout << "R2=" << r2metric << endl;
   
    // Освобождение памяти
    // Finally free memory
    safe_xgboost(XGBoosterFree(booster));
    safe_xgboost(XGDMatrixFree(dtrain));
    safe_xgboost(XGDMatrixFree(dtest));

    return 0;
}
