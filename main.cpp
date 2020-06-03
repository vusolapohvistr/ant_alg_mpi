#include <stdlib.h>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>


#include "Ant.h"

#define GRAPH_NODES 10000

auto get_mat_from_file(std::string file_path) -> std::vector<std::vector<double_t>>;
auto gen_graph(int size, int additional_edges, double_t max_weight) -> std::vector<std::vector<int>>;

void vaporize_pheromones(std::vector<std::vector<int>> &pheromones_mat, Config &config) {
    for (auto &line : pheromones_mat) {
        for (auto &val : line) {
            int new_val = val * (1 - config.ro);
            if (new_val == 0) {
                new_val = 1;
            }
            val = new_val;
        }
    }
}

template <class T>
void print_vector(std::vector<T> &vec) {
    for (auto& val : vec) {
        std::cout << val << " ";
    }
}

int main(int argc, char *argv[]) {
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    char  (*all_proc_names)[MPI_MAX_PROCESSOR_NAME];
    int numprocs;
    int MyID;
    int namelen;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
    MPI_Get_processor_name(processor_name, &namelen);

    Config config = Config {
            .alfa = 0.7,
            .beta = 0.3,
            .ant_capacity = 1000.0,
            .ro = 0.3,
            .ant_num = 30,
            .iters = 100
    };

    int ants_per_worker = config.ant_num / numprocs;
    int ants_num = ants_per_worker * (numprocs - 1);
    int32_t start_point = 0;
    std::vector<int> targets({0, 2, 4});

    if (MyID == 0) {
        std::vector<int> answer;
        auto min_way = std::numeric_limits<int>::max();

        std::cout << "main process" << std::endl;

        auto weight_mat = gen_graph(GRAPH_NODES, 100, 1000.0);

        // Sending weight mat
        for (int i = 1; i < numprocs; i++) {
            for (int j = 0; j < GRAPH_NODES; j++) {
                MPI_Send(&weight_mat[j][0], GRAPH_NODES, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        auto pheromones_mat = std::vector<std::vector<int>> (GRAPH_NODES, std::vector<int> (GRAPH_NODES, 1));

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < config.iters; i++) {

            // Sending pheromones mat
            for (int i = 1; i < numprocs; i++) {
                for (int j = 0; j < GRAPH_NODES; j++) {
                    MPI_Send(&pheromones_mat[j][0], GRAPH_NODES, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }

            // Receiving ants paths
            for (int i = 0; i < ants_num; i++) {
                // Receive ant's result
                int path_length = 0;
                MPI_Recv(&path_length, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int total_way = 0;
                MPI_Recv(&total_way, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<int> path(path_length);
                MPI_Recv(&path[0], path_length, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Changing pheromones matrix and min way
                auto ant = Ant(&config);
                ant.path = path;
                ant.total_way = total_way;
                if (ant.total_way < min_way) {
                    min_way = ant.total_way;
                    answer = path;
                }
                ant.change_pheromones_mat(pheromones_mat);
            }

            vaporize_pheromones(pheromones_mat, config);
        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Time spent "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
                  << "ms" << std::endl;
        std::cout << "Total way " << min_way << std::endl;
        std::cout << "Path: ";
        print_vector(answer);

    } else {
        std::cout << "worker" << std::endl;

        // Receive weight matrix
        std::vector<std::vector<int>> weight_mat(GRAPH_NODES, std::vector<int> (GRAPH_NODES));
        for (int i = 0; i < GRAPH_NODES; i++) {
            MPI_Recv(&weight_mat[i][0], GRAPH_NODES, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Main ants loop
        auto pheromones_mat = std::vector<std::vector<int>> (GRAPH_NODES, std::vector<int> (GRAPH_NODES, 1));
        for (int i = 0; i < config.iters; i++) {

            // Receive pheromones mat
            for (int i = 0; i < GRAPH_NODES; i++) {
                MPI_Recv(&pheromones_mat[i][0], GRAPH_NODES, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Creating ants and sending their paths to main thread
            for (int i = 0; i < ants_per_worker; i++) {
                auto ant = Ant(&config);
                ant.go(start_point, targets, weight_mat, pheromones_mat);

                // Sending ant result
                int path_length = ant.path.size();
                int total_way = ant.total_way;
                MPI_Send(&path_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&total_way, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                MPI_Send(&ant.path[0], path_length, MPI_INT, 0, 2, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
}

auto get_mat_from_file(std::string file_path) -> std::vector<std::vector<double_t>> {
    std::vector<std::vector<double_t>> answer;
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        throw new FILE;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double_t> row;
        while (iss) {
            double_t a;
            iss >> a;
            row.push_back(a);
        }
        answer.push_back(row);
    }

    return answer;
}

auto gen_graph(int size, int additional_edges, double_t max_weight) -> std::vector<std::vector<int>> {
    auto result = std::vector<std::vector<int>> (size, std::vector<int> (size, 0.0));
    std::set<int> used_nodes;

    std::default_random_engine generator;
    std::uniform_real_distribution<double_t> uniform_double(0.0 , max_weight);
    std::uniform_int_distribution<int> uniform_int(0, size - 1);

    auto current_node = 0;

    while (used_nodes.size() < size) {
        auto next_node = current_node;
        while (next_node == current_node) {
            next_node = uniform_int(generator);
        }
        auto weight = (int) uniform_double(generator);
        result[current_node][next_node] = weight;
        result[next_node][current_node] = weight;
        used_nodes.insert(current_node);
        current_node = next_node;
    }

    if (current_node != 0) {
        result[current_node][0] = max_weight / 2;
        result[0][current_node] = max_weight / 2;
    }

    for (int i = 0; i < additional_edges; i++) {
        auto next_node = current_node;
        while (next_node == current_node) {
            next_node = uniform_int(generator);
        }
        auto weight = (int) uniform_double(generator);
        result[current_node][next_node] = weight;
        result[next_node][current_node] = weight;
        used_nodes.insert(current_node);
        current_node = next_node;
    }

    if (current_node != 0) {
        result[current_node][0] = max_weight / 2;
        result[0][current_node] = max_weight / 2;
    }

    return result;
}