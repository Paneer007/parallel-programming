#include <stddef.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>

extern "C++"
{
#include "engine.hpp"
}

extern "C++"
{
    // Allocated Unified memory for a Value Object
    void allocValue(Value **v, size_t num)
    {
        cudaError_t err = cudaMallocManaged(v, num * sizeof(Value));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed allocating unified memory for Value: %s\n", cudaGetErrorString(err));
            // Handle the error appropriately
            exit(1);
        }
    }

    // Allocated Unified memory for an array of Value Object
    void allocValueArr(Value ***ptr, size_t len)
    {
        cudaError_t err = cudaMallocManaged(ptr, len * sizeof(Value *));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed allocating unified memory for an array of Values: %s\n", cudaGetErrorString(err));
            // Handle the error appropriately
            exit(1);
        }
    }

    // Backpropagation of gradients in computation graph

    void add_gradient(Value *v)
    {
        for (int i = 0; i < v->n_children; i++)
        {
            v->children[i]->grad += v->grad;
        }
    }

    void sub_gradient(Value *v)
    {
        v->children[0]->grad += v->grad;
        v->children[1]->grad -= v->grad;
    }

    void mul_gradient(Value *v)
    {
        v->children[0]->grad += v->children[1]->val * v->grad;
        v->children[1]->grad += v->children[0]->val * v->grad;
    }

    void div_gradient(Value *v)
    {
        v->children[0]->grad += (1.0 / v->children[1]->val) * v->grad;
        v->children[1]->grad += (-v->children[0]->val / (v->children[1]->val * v->children[1]->val)) * v->grad;
    }

    void power_gradient(Value *v)
    {
        v->children[0]->grad += (v->children[1]->val * pow(v->children[0]->val, v->children[1]->val - 1)) * v->grad;
        if (v->children[0]->val > 0)
        { // Ensure base is positive before computing log
            v->children[1]->grad += (log(v->children[0]->val) * pow(v->children[0]->val, v->children[1]->val)) * v->grad;
        }
    }

    void Value::backward()
    {
        Value *topo[5000]; // Assuming a maximum of 10000 nodes in the computation graph for simplicity
        int topo_size = 0;
        Value *visited[5000];
        int visited_size = 0;

        // Obtain topological order of computation graph
        this->build_topo(topo, &topo_size, visited, &visited_size);

        this->grad = 1.0; // Initialise the first node gradient

        // Traverse the node backwards in topological order
        // and update the gradient value of children nodes
        for (int i = topo_size - 1; i >= 0; --i)
        {
            Value *v = topo[i];
            switch (v->op)
            {
            case ADD:
                add_gradient(v);
                break;
            case SUB:
                sub_gradient(v);
                break;
            case MUL:
                mul_gradient(v);
                break;
            case DIV:
                div_gradient(v);
                break;
            case POW:
                power_gradient(v);
                break;
            default:
                break;
            }
        }
    }

    void free_value(Value *v)
    {
        if (v->children)
        {
            cudaFree(v->children);
        }
        cudaFree(v);
    }
}