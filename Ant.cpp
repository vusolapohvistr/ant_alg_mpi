//
// Created by news on 27.05.2020.
//

#include "Ant.h"

Ant::Ant(Config* conf) {
    config = conf;
}

void Ant::print_config() {
    std::cout << config->alfa;
}

void Ant::go(int32_t start_point,
        std::vector<int> &targets,
        std::vector<std::vector<int>> &weight_mat,
        std::vector<std::vector<int>> &pheromones_mat) {
    auto current_pos = start_point;
    path.push_back(current_pos);

    auto unvisited_targets = targets;

    std::default_random_engine generator;
    std::uniform_real_distribution<double_t> uniform(0.0 , 1.0);

    while (unvisited_targets.size() > 0 || (unvisited_targets.empty() && path.back() != start_point)) {
        auto balance_sum = 0.0;
        for (int i = 0; i < weight_mat.size(); i++) {
            if (std::abs(weight_mat[current_pos][i]) < DBL_EPSILON) {
                continue;
            }
            balance_sum += std::pow(pheromones_mat[current_pos][i], config->alfa)
                    * (1.0 / std::pow(weight_mat[current_pos][i], config->beta));
        }

        if (balance_sum == 0.0) {
            cant_find_way = true;
            break;
        }

        for (int i = 0; i < weight_mat.size(); i++) {
            if (std::abs(weight_mat[current_pos][i]) < DBL_EPSILON || (path.size() > 1 && i == path[path.size() - 2])) {
                continue;
            }
            auto prob_to_move = std::pow(pheromones_mat[current_pos][i], config->alfa)
                                * (1.0 / std::pow(weight_mat[current_pos][i], config->beta))
                                / balance_sum;

            auto random_number = uniform(generator);
            if (random_number < prob_to_move) {
                path.push_back(i);
                auto unvisited_target = std::find(unvisited_targets.begin(), unvisited_targets.end(), i);
                if (unvisited_target != unvisited_targets.end()) {
                    unvisited_targets.erase(unvisited_target);
                }
                total_way += weight_mat[current_pos][i];
                current_pos = i;
                break;
            }
        }
    }
}

void Ant::change_pheromones_mat(std::vector<std::vector<int>> pheromones_mat) {
    if (!cant_find_way) {
        for (int i = 0; i < path.size() - 2; i++) {
            pheromones_mat[path[i]][path[i + 1]] +=
                    config->ant_capacity / total_way;
        }
    }
    total_way = 0.0;
    path.clear();
}

