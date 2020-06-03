//
// Created by news on 27.05.2020.
//

#include "AntAlg.h"

void vaporize_pheromones(std::vector<std::vector<int>> &pheromones_mat, Config* config) {
    for (auto &line : pheromones_mat) {
        for (auto &val : line) {
            val *= 1 - config->ro;
        }
    }
}

AntAlgResult ant_alg_sync(std::vector<std::vector<int>> &weight_mat, Config *config, int start_point, std::vector<int> &targets) {
    std::vector<int> answer;
    auto min_way = std::numeric_limits<int>::max();
    auto pheromones_mat = std::vector<std::vector<int>> (weight_mat.size(), std::vector<int> (weight_mat.size(), 1.0));
    auto ants = std::vector<Ant> (config->ant_num, Ant(config));

    for (int i = 0; i < config->iters; i++) {
        for (auto &ant : ants) {
            ant.go(start_point, targets, weight_mat, pheromones_mat);
            if (ant.total_way < min_way) {
                min_way = ant.total_way;
                answer = ant.path;
            }
        }
        vaporize_pheromones(pheromones_mat, config);
        for (auto &ant : ants) {
            ant.change_pheromones_mat(pheromones_mat);
        }
    }
    return AntAlgResult {answer, min_way};
}

AntAlgResult ant_alg_openmp(std::vector<std::vector<int>> &weight_mat, Config *config, int start_point,
                            std::vector<int> &targets) {
    std::vector<int> answer;
    auto min_way = std::numeric_limits<int>::max();
    auto pheromones_mat = std::vector<std::vector<int>> (weight_mat.size(), std::vector<int> (weight_mat.size(), 1.0));
    auto ants = std::vector<Ant> (config->ant_num, Ant(config));

    for (int i = 0; i < config->iters; i++) {
#pragma omp parallel for
        for (int i = 0; i < ants.size(); i++)
        {
            auto ant = &ants[i];
            ant->go(start_point, targets, weight_mat, pheromones_mat);
            if (ant->total_way < min_way) {
                min_way = ant->total_way;
                answer = ant->path;
            }
        }
        vaporize_pheromones(pheromones_mat, config);
        for (auto &ant : ants) {
            ant.change_pheromones_mat(pheromones_mat);
        }
    }
    return AntAlgResult {answer, min_way};
}