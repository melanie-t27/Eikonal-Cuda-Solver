#ifndef EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH
#define EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH

#include "Mesh.cuh"
#include "Kernels.cu"
#include <cuda_runtime.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <functional>
#include <thrust/async/scan.h>

constexpr int NUM_THREADS = 1024;
constexpr int SIZE_WARP = 32;

template <int D,typename Float>
class Solver {
public:
    Solver(Mesh<D, Float>* mesh) : mesh(mesh){
        gpuDataTransfer();
    }

    // error handling
    static void cudaCheck(std::string mess) {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cout << "cuda err = " << err << std::endl;
            std::cout << "mess = "<< mess << std::endl;

            exit(1);
        }
    }

    // we transfer the necessary data from host to device
    void gpuDataTransfer(){
        // Allocate memory in device
        cudaMalloc(&geo_dev, sizeof(Float) * mesh->getNumberVertices() * D);
        cudaMalloc(&tetra_dev, sizeof(int) * mesh->get_tetra().size());
        cudaMalloc(&shapes_dev, sizeof(TetraConfig) * mesh->getShapes().size());
        cudaMalloc(&ngh_dev, sizeof(int) * mesh->get_ngh().size());
        cudaMalloc(&M_dev, sizeof(Float) * mesh->get_M().size());
        // Copy data from host to device
        cudaMemcpy(geo_dev, mesh->getGeo().data(), sizeof(Float) * mesh->getNumberVertices() * D, cudaMemcpyHostToDevice);
        cudaMemcpy(tetra_dev, mesh->get_tetra().data(), sizeof(int) * mesh->get_tetra().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(shapes_dev, mesh->getShapes().data(), sizeof(TetraConfig) * mesh->getShapes().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ngh_dev, mesh->get_ngh().data(), sizeof(int) * mesh->get_ngh().size(), cudaMemcpyHostToDevice);
        cudaMemcpy(M_dev, mesh->get_M().data(), sizeof(Float) * mesh->get_M().size(), cudaMemcpyHostToDevice);
    }

    // note: PartitionsVertices is an array such that (assuming vertices are clustered according to their subdomain, i.e. )
    // 1)length(PartitionsVertices) == number of subdomains in the mesh
    // 2)PartitionsVertices[i] == k <==> vertices from PartitionsVertices[i-1](inclusive)
    // to PartitionsVertices[i](exclusive) belong to subdomains i. If i==0,
    // then PartitionsVertices[-1] is assumed to be equals to 0, Moreover PartitionsVertices[i],
    // with 0 <= i < mesh->getPartitionsNumber() - 1, is the index of the first node belonging to subdomain i+1

    void solve(std::vector<int> source_nodes, Float tol, Float infinity_value, const std::string& output_file_name){
        using VectorExt = typename CudaEikonalTraits<Float, D>::VectorExt;
        using VectorV = typename CudaEikonalTraits<Float, D>::VectorV;
        using Matrix = typename CudaEikonalTraits<Float, D>::Matrix;

        std::vector<cudaStream_t> streams;
        streams.resize(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> sAddrs(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> cLists(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> nbhNrs(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<TetraConfig>> elemLists(mesh->getPartitionsNumber());
        std::vector<thrust::device_vector<int>> predicate(mesh->getPartitionsNumber());
        thrust::device_vector<size_t> elemListSizes(mesh->getPartitionsNumber());
        thrust::device_vector<int> partitions_boundaries(mesh->getPartitionsBoundaries());
        thrust::device_vector<Float> solutions(mesh->getNumberVertices() * mesh->getPartitionsNumber(), infinity_value);
        thrust::device_vector<int> active_lists_dev(mesh->getNumberVertices() * mesh->getPartitionsNumber(), 0);


        #pragma omp parallel for num_threads(mesh->getPartitionsNumber())
        for(int i = 0; i < mesh->getPartitionsNumber(); i++) {
            size_t vec_size = getDomainSize(i);
            sAddrs[i] = thrust::device_vector<int>(vec_size);
            cLists[i] = thrust::device_vector<int>(vec_size);
            nbhNrs[i] = thrust::device_vector<int>(vec_size);
            predicate[i] = thrust::device_vector<int>(mesh->getNumberVertices(), 0);
            elemLists[i] = thrust::device_vector<TetraConfig>(getTetraNumberinDomain(i));
        }

        for(int s : source_nodes) {
            for(int i = 0; i < mesh->getPartitionsNumber();i++) {
                solutions[i * mesh->getNumberVertices() + s] = 0.0;
            }
        }

        setup(source_nodes, infinity_value, active_lists_dev);

        // create streams, one for each domain
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamCreate(&streams[i]);
        }

        bool not_converged = true;
        int maximum_neighbour_tetra = mesh->get_maximum_neighbour_tetra();
        std::vector<Float> solutions_host(mesh->getNumberVertices());
        #pragma omp parallel default(shared) num_threads(mesh->getPartitionsNumber())
        {
            while(not_converged) {
                #pragma omp single
                {
                    not_converged = false;  
                }
                
                #pragma omp for
                for(int domain = 0; domain < mesh->getPartitionsNumber(); domain++) {
                    size_t domain_size = getDomainSize(domain);
                    size_t begin_domain = getBeginDomain(domain);
                    size_t end_domain = begin_domain + domain_size;
                    int numBlocks = domain_size / NUM_THREADS + 1;
                    // we count the number of active nodes (value set to 1 in active_list)
                    int active_nodes = thrust::count(thrust::cuda::par_nosync.on(streams[domain]), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices()]), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices() + domain_size]) , 1);
                    if(active_nodes != 0) {
                        #pragma omp critical
                        {
                            not_converged = true;
                        }
                        
                    }
                    while(active_nodes > 0) {
                        // we perform exclusive scan and result will be stored in sAddrs
                        thrust::exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices()]), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices() + domain_size]), sAddrs[domain].begin());//, unary_op, 0, binary_op);  //sAddrs may be shorter
                        // we perform compact and store in cLists, which will be the compact active list
                        compact<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices()]), domain_size ,thrust::raw_pointer_cast(cLists[domain].data()), begin_domain);
                        // we count the total number of neighbouring tetrahedra for each node in the list and store in nbhNrs
                        count_Nbhs<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(cLists[domain].data()), ngh_dev, thrust::raw_pointer_cast(nbhNrs[domain].data()), active_nodes, mesh->getNumberVertices() ,mesh->getShapes().size());
                        // we perform exclusive scan to get the addresses and store in aAddrs
                        thrust::async::exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), nbhNrs[domain].begin(), nbhNrs[domain].begin() + active_nodes, sAddrs[domain].begin());
                        // we perform a gather and store all the neighbouring tetrahedra for each node in cLists in elemLists, based on the address generated in previous line

                        gather_elements<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(cLists[domain].data()), thrust::raw_pointer_cast(nbhNrs[domain].data()),
                                                                                        thrust::raw_pointer_cast(elemLists[domain].data()), active_nodes, thrust::raw_pointer_cast(&elemListSizes[domain]), ngh_dev, shapes_dev);
                        // construct predicate
                        constructPredicate<D, Float><<<active_nodes, maximum_neighbour_tetra, 0, streams[domain]>>>(thrust::raw_pointer_cast(elemLists[domain].data()), thrust::raw_pointer_cast(&elemListSizes[domain]), active_nodes, thrust::raw_pointer_cast(sAddrs[domain].data()), tetra_dev, geo_dev, thrust::raw_pointer_cast(&solutions[domain*mesh->getNumberVertices()]), thrust::raw_pointer_cast(predicate[domain].data()), M_dev, tol, thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices()]), begin_domain, domain_size);
                        //started_blocks[domain] += active_nodes;
                        //count of the current domain activated nodes, other will be processed by other domains in a successive iteration
                        int active_neighbors_node = thrust::count(thrust::cuda::par_nosync.on(streams[domain]), predicate[domain].begin() + begin_domain, predicate[domain].begin() + end_domain, 1);
                        if(active_neighbors_node != 0) {
                            //started_blocks[domain] += active_neighbors_node;

                            // we perform exclusive scan on predicate[domain] and store the result in sAddrs
                            thrust::async::exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), predicate[domain].begin() + begin_domain, predicate[domain].begin() + end_domain, sAddrs[domain].begin());
                            // compact and store in cLists
                            compact<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(predicate[domain].data()) + begin_domain, domain_size,thrust::raw_pointer_cast(cLists[domain].data()), begin_domain);
                            // count the number of neighbouring tetrahedra
                            count_Nbhs<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(cLists[domain].data()), ngh_dev, thrust::raw_pointer_cast(nbhNrs[domain].data()), active_neighbors_node, mesh->getNumberVertices() ,mesh->getShapes().size());
                            // perform scan to get addresses to store in sAddrs
                            thrust::async::exclusive_scan(thrust::cuda::par_nosync.on(streams[domain]), nbhNrs[domain].begin(), nbhNrs[domain].begin() + active_neighbors_node, sAddrs[domain].begin());
                            gather_elements<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(sAddrs[domain].data()), thrust::raw_pointer_cast(cLists[domain].data()), thrust::raw_pointer_cast(nbhNrs[domain].data()),
                                                                                        thrust::raw_pointer_cast(elemLists[domain].data()), active_neighbors_node, thrust::raw_pointer_cast(&elemListSizes[domain]), ngh_dev, shapes_dev);
                            processNodes<D, Float><<<active_neighbors_node, maximum_neighbour_tetra, 0, streams[domain]>>>(thrust::raw_pointer_cast(elemLists[domain].data()),thrust::raw_pointer_cast(&elemListSizes[domain]),active_neighbors_node, thrust::raw_pointer_cast(sAddrs[domain].data()),tetra_dev, geo_dev, thrust::raw_pointer_cast(&solutions[domain * mesh->getNumberVertices()]), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices()]), M_dev, tol, begin_domain);
                        }
                        removeConvergedNodes<<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices()]), domain_size);
                        active_nodes = thrust::count(thrust::cuda::par_nosync.on(streams[domain]), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices()]), thrust::raw_pointer_cast(&active_lists_dev[domain*mesh->getNumberVertices() + domain_size]),  1);
                        // set to 0 predicate[domain]
                        thrust::fill(thrust::cuda::par_nosync.on(streams[domain]), predicate[domain].begin(), predicate[domain].end(), 0);
                    }
                }
                    cudaDeviceSynchronize();   


                // propagate predicate
                
                    #pragma omp for
                    for(int domain = 0; domain < mesh->getPartitionsNumber() && not_converged; domain++) {
                        size_t domain_size = getDomainSize(domain);
                        size_t begin_domain = getBeginDomain(domain);
                        int numBlocks = domain_size / NUM_THREADS + 1;
                        propagateSolution<Float><<<numBlocks, NUM_THREADS, 0, streams[domain]>>>(thrust::raw_pointer_cast(&active_lists_dev[0]), domain, domain_size, begin_domain, mesh->getNumberVertices(), thrust::raw_pointer_cast(partitions_boundaries.data()), thrust::raw_pointer_cast(solutions.data()));
                    }
                        
                        cudaDeviceSynchronize();
                        

                }
            }
        

        // copy solutions back to host
        for(int i = 0; i < mesh->getPartitionsNumber();i++) {
            int begin = getBeginDomain(i);
            int size = getDomainSize(i);
            cudaMemcpy(solutions_host.data() + begin, thrust::raw_pointer_cast(solutions.data() + i*mesh->getNumberVertices()) + begin, sizeof(Float) * size, cudaMemcpyDeviceToHost);
        }
        
        // write solution to file
        mesh->getSolutionsVTK(output_file_name, solutions_host);

        // destroy streams
        for(int i = 0; i < mesh->getPartitionsNumber(); i++){
            cudaStreamDestroy(streams[i]);
        }

    }

    ~Solver() {
        cudaFree(geo_dev);
        cudaFree(tetra_dev);
        cudaFree(shapes_dev);
        cudaFree(ngh_dev);
        cudaFree(M_dev);
    }



private:
    Mesh<D, Float>* mesh;
    Float* geo_dev;
    int* tetra_dev;
    TetraConfig* shapes_dev;
    int* ngh_dev;
    Float* M_dev;

    // method used to set the solutions to infinity value, set to 0 the solutions on source nodes,
    // and set to 1 the active nodes in active_list, then copy data from active_list to active_list_dev
    void setup(std::vector<int>& source_nodes, Float infinity_value, thrust::device_vector<int>& active_lists_dev){
        // we initialize the vector of solutions to infinity value, and set to 0 the solution on source nodes
        for(auto source : source_nodes) {
            for(int i = mesh->get_ngh()[source]; i < ((source != mesh->getNumberVertices() - 1) ? mesh->get_ngh()[source+1] : mesh->getShapes().size()); i++) {
                for(int j = 0; j < D +1; j++) {
                    int v = mesh->get_tetra()[(D+1)*mesh->getShapes()[i].tetra_index + j];
                    if(v != source) {
                        // if the node is not a source node, we set in the corresponding position in active_list value= 1
                        int domain = getDomain(v);
                        active_lists_dev[domain*mesh->getNumberVertices() + v - getBeginDomain(domain)] = 1;
                    }
                }
            }
        }
    }

    // method, that provided a vertex (index) returns the domain it belongs to
    int getDomain(int vertex_index) {
        for(int i = 0; i < mesh->getPartitionsNumber(); i++) {
            if(vertex_index < mesh->getPartitionVertices()[i]) {
                return i;
            }
        }
        return -1;
    }

    //requires domain lies in range [0, mesh->getPartitionsNumber())
    //requires ngh is an array such that ngh[i] is the index in msh->shapes of the first tetrahedra near node i
    //ensures that return value is number of tetrahedra id domain 'domain', satisfied as
    //1)end + 1 is the index in mesh->shapes of the first tetrahedra near node PartitionsVertices[domain] -> is also the index in shapes of the first tetrahedra belonging to domain 'domain + 1'. Therefore end is the index of the last tetrahedra (in shapes) belonging to domain 'domain'(inclusive)
    //2) start, similarly to point 1) is the last tetrahedra belonging to domain 'domain-1'(inclusive)
    //therefore end - start is actually what is meant to be
    int getTetraNumberinDomain(int domain) {
        int end = (domain != mesh->getPartitionsNumber() - 1) ? (mesh->get_ngh()[getBeginDomain(domain + 1)] -1) : mesh->getShapes().size() - 1;
        int start = mesh->get_ngh()[getBeginDomain(domain)] - 1;
        return end - start;
    }


    //requires domain lies in range
    int getBeginDomain(int domain) {
        return (domain != 0) ? mesh->getPartitionVertices()[domain - 1] : 0;
    }


    int getDomainSize(int domain) {
        return getBeginDomain(domain+1) - getBeginDomain(domain);
    }

};

#endif //EIKONAL_CUDA_CESARONI_TONARELLI_TRABACCHIN_SOLVER_CUH