#ifndef PARSE_H
#define PARSE_H

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Utils.h"

// Graph class to parse TSP files

class Graph {
public:
  std::string name;
  std::string type;
  bool in_gpu;
  enum { EUC_2D, CEIL_2D, GEO } edge_type;
  int N; // size of the graph
  float *distances;
  float *gpu_distances;

  Graph(std::string &filename);

  float nearest_neigh() {
    float total_distance = 0;
    int start = 0;
    for (int i = 0; i < N; i++) {
      float distance_min = MAXFLOAT;
      int idx_min = 0;
      for (int j = 0; j < N; j++) {
        if (distances[start * N + j] < distance_min && start != j) {
          distance_min = distances[start * N + j];
          idx_min = j;
        }
      }
      start = idx_min;
      total_distance += distance_min;
    }
    return total_distance;
  }

  void to_gpu() {
    if (in_gpu) {
      return;
    }
    gpuErrchk(cudaMalloc((void **)&gpu_distances, N * N * sizeof(float)));
    gpuErrchk(cudaMemcpy(gpu_distances, distances, N * N * sizeof(float),
                         cudaMemcpyHostToDevice));
    in_gpu = true;
  }
  ~Graph() {
    delete[] distances;
    if (in_gpu) {
      gpuErrchk(cudaFree(gpu_distances));
    }
  }
};

Graph::Graph(std::string &filename) {
  in_gpu = false;
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }
  std::string line;
  bool node_section = false;

  float *x_arr, *y_arr;
  int n = 0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string key;
    std::string dummy;
    iss >> key;
    if (key == "NAME" || key == "NAME:") {
      iss >> dummy; // Skip the :
      if (dummy == ":") {
        iss >> dummy;
      }
      this->name = dummy;
    } else if (key == "TYPE" || key == "TYPE:") {
      iss >> dummy; // Skip the :
      if (dummy == ":") {
        iss >> dummy;
      }
      this->type = dummy;
    } else if (key == "DIMENSION" || key == "DIMENSION:") {
      iss >> dummy; // Skip the :
      if (dummy == ":") {
        iss >> dummy;
      }
      this->N = stoi(dummy);
    } else if (key == "EDGE_WEIGHT_TYPE" || key == "EDGE_WEIGHT_TYPE:") {
      std::string edge_type_str;
      iss >> edge_type_str; // Skip the :
      if (edge_type_str == ":") {
        iss >> edge_type_str;
      }
      if (edge_type_str == "EUC_2D")
        this->edge_type = EUC_2D;
      else if (edge_type_str == "CEIL_2D")
        this->edge_type = CEIL_2D;
      else if (edge_type_str == "GEO")
        this->edge_type = GEO;
      else
        throw std::runtime_error("Unknown edge weight type");
    } else if (key == "NODE_COORD_SECTION") {
      assert(this->N > 0);
      node_section = true;
      distances = new float[this->N * this->N];
      x_arr = new float[this->N];
      y_arr = new float[this->N];
    } else if (key == "EOF") {
      break;
    } else if (node_section) {
      n++;
      std::istringstream node(line);
      int node_id;
      float x, y;
      node >> node_id >> x >> y;
      x_arr[node_id - 1] = x;
      y_arr[node_id - 1] = y;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        distances[i * N + j] = 0;
      } else {
        if (edge_type == EUC_2D) {
          float dx = x_arr[i] - x_arr[j];
          float dy = y_arr[i] - y_arr[j];
          distances[i * N + j] = sqrt(dx * dx + dy * dy);
        } else if (edge_type == CEIL_2D) {
          float dx = x_arr[i] - x_arr[j];
          float dy = y_arr[i] - y_arr[j];
          distances[i * N + j] = std::ceil(sqrt(dx * dx + dy * dy));
        } else {
          float latitude1 = ConvertToRadian(x_arr[i]);
          float longitude1 = ConvertToRadian(y_arr[i]);
          float latitude2 = ConvertToRadian(x_arr[j]);
          float longitude2 = ConvertToRadian(y_arr[j]);

          float q1 = cos(longitude1 - longitude2);
          float q2 = cos(latitude1 - latitude2);
          float q3 = cos(latitude1 + latitude2);
          distances[i * N + j] =
              6378.388 * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0;
        }
      }
    }
  }
  delete[] x_arr;
  delete[] y_arr;
}

void save_output(std::string &name,
                 std::pair<float, std::vector<int>> &result) {
  std::ofstream file(name);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file");
  }
  file << result.first << std::endl;
  for (auto value : result.second) {
    file << value + 1 << " ";
  }
  file << std::endl;
  file.close();
}

#endif
