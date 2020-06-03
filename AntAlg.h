//
// Created by news on 27.05.2020.
//

#include "Ant.h"

#ifndef ANT_ALG_OPENMP_ANTALG_H
#define ANT_ALG_OPENMP_ANTALG_H

struct AntAlgResult {
    std::vector<int> path;
    int total_way;
};

void vaporize_pheromones(std::vector<std::vector<int>> &pheromones_mat, Config* config);

AntAlgResult ant_alg_sync(std::vector<std::vector<int>> &weight_mat, Config* config, int start_point, std::vector<int> &targets);

AntAlgResult ant_alg_openmp(std::vector<std::vector<int>> &weight_mat, Config* config, int start_point, std::vector<int> &targets);

#endif //ANT_ALG_OPENMP_ANTALG_H
