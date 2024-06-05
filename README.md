## A simple test C++ program for XGBoost C API. Testing dataset is Boston Housing.

 
### 1. Download XGBoost https://github.com/dmlc/xgboost and build it

### 2. Then git clone this repository, create CMakeLists.txt inside the folder:

  cmake_minimum_required(VERSION 3.0.0)
  project(xgboostproj VERSION 0.1.0 LANGUAGES C CXX)

  add_executable(xgboost_cpp_boston main.cpp)

  include_directories(/<your xgboost source folder>)
  target_link_libraries(xgboost_cpp_boston <your xgboost lib folder>)

### 3. Build the program:
  md Build
  cd Build
  cmake ..
  make
### 4. And launch  
  ./xgboost_cpp_boston

