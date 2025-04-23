#include "Core/AntThread.h"
#include "Core/QueenAnt.h"
#include "GraphUtils/Graph.h"

#include <cassert>
#include <string>

int main(int argc, const char *argv[]) {
  assert(argc == 9);
  std::string inputFile = argv[1];
  std::string outputFile = argv[2];
  std::string type = argv[3];
  int num_iter = std::stoi(argv[4]);
  double alpha = std::stod(argv[5]);
  double beta = std::stod(argv[6]);
  double evaporate = std::stod(argv[7]);
  unsigned long seed = std::stol(argv[8]);
  auto input = Graph(inputFile);

  input.to_gpu();
  if (type == "WORKER") {
    auto solution = AntThread(input, num_iter, alpha, beta, evaporate, seed);
    save_output(outputFile, solution);
  } else if (type == "QUEEN") {
    auto solution = QueenAnt(input, num_iter, alpha, beta, evaporate, seed);
    save_output(outputFile, solution);
  } else {
    throw std::invalid_argument("Invalid type");
  }
  return 0;
}