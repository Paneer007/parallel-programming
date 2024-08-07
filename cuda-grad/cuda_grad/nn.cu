#include <stddef.h>
#include <stdio.h>
#include <cuda.h>

extern "C++"
{
#include "nn.hpp"
#include "engine.hpp"
}

extern "C++"
{
    // Allocate unified memory for a neuron
    void allocNeuron(Neuron **n, size_t num)
    {
        auto err = cudaMallocManaged(n, num * sizeof(Neuron));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed allocating unified memory for Neuron: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }

    // Allocate unified memory for an array of neuron
    void allocNeuronArr(Neuron ***ptr, size_t len)
    {
        auto err = cudaMallocManaged(ptr, len * sizeof(Neuron *));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed allocating unified memory for an array of Neurons: %s\n", cudaGetErrorString(err));
            exit(1);
        }
    }

    // Init a neuron using unified memory
    Neuron *init_neuron(int num_in)
    {
        Neuron *neuron;
        allocNeuron(&neuron, 1);
        allocValueArr(&(neuron->w), num_in);
        for (int i = 0; i < num_in; i++)
        {
            neuron->w[i] = init_value((rand() % 2000 - 1000) / 1000.0); // random values between -1 and 1
        }
        neuron->b = init_value(0);
        neuron->num_in = num_in;
        return neuron;
    }

    // Init a layer of neurons using unified memory
    Layer *init_layer(int num_in, int num_out)
    {
        Layer *layer;
        cudaMallocManaged(&layer, sizeof(Layer));
        allocNeuronArr(&(layer->neurons), num_out);
        for (int i = 0; i < num_out; i++)
        {
            // Init neurons
            layer->neurons[i] = init_neuron(num_in);
        }
        layer->num_out = num_out;
        return layer;
    }

    // Init a MLP using unified memory
    MLP *init_mlp(int *sizes, int num_layers)
    {
        MLP *mlp;
        cudaMallocManaged(&mlp, sizeof(MLP));
        cudaMallocManaged(&(mlp->layers), (num_layers - 1) * sizeof(Layer *));
        for (int i = 0; i < num_layers - 1; i++)
        {
            mlp->layers[i] = init_layer(sizes[i], sizes[i + 1]);
        }
        mlp->num_layers = num_layers - 1;
        return mlp;
    }

    // Multiply weights and values and store in resultant value
    __device__ void mul_dev(Value *w, Value *x, Value *v)
    {
        v->val = w->val * x->val;
        v->grad = 0;
        v->children[0] = w;
        v->children[1] = x;
        v->n_children = 2;
        v->op = MUL;
    }

    // Performs a forward pass through the neural network
    __global__ void layer_forward(Layer *layer, Value **x, Value **out, Value **products)
    {
        // Obtain node to be processed
        auto datapoint_id = blockIdx.y;
        auto neuron_idx = blockIdx.x;
        auto n = layer->neurons[neuron_idx];

        // Calculate the indices required
        auto input_idx = blockDim.x * blockIdx.y + threadIdx.x;
        auto prod_idx = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
        auto out_idx = datapoint_id * gridDim.x + neuron_idx;
        auto prod = products[prod_idx];

        // Compute product and store intermediate result
        mul_dev(n->w[threadIdx.x], x[input_idx], prod);
        out[out_idx]->children[threadIdx.x] = prod;

        // Update output value atomically
        atomicAdd(&(out[out_idx]->val), prod->val);
        __syncthreads();

        // Add bias and complete output value for the last thread in the block
        if (threadIdx.x == blockDim.x - 1)
        {
            out[out_idx]->val += n->b->val;
            out[out_idx]->children[blockDim.x] = n->b;
        }
    }

    // Performs forward pass through a MLP
    Value **MLP::mlp_forward(Value **x, int num_in)
    {
        // Loop through each layer of the MLP
        for (int i = 0; i < this->num_layers; i++)
        {
            auto curr_layer = this->layers[i];

            // Initialize output values for the current layer
            float initialSums[curr_layer->num_out];
            memset(initialSums, 0.0, curr_layer->num_out * sizeof(float));
            auto out = init_values(initialSums, curr_layer->num_out);

            // Allocate and initialize output children for each neuron:
            for (int i = 0; i < curr_layer->num_out; i++)
            {
                allocValueArr(&(out[i]->children), num_in);
                out[i]->n_children = num_in;
                out[i]->op = ADD;
            }

            // Allocate and initialize intermediate products
            Value **products;
            allocValueArr(&products, num_in * curr_layer->num_out);
            for (int i = 0; i < num_in * curr_layer->num_out; i++)
            {
                products[i] = init_value(0);
                allocValueArr(&(products[i]->children), 2);
            }

            // Define grid size and launch the layer_forward kernel
            dim3 grid_size(curr_layer->num_out, 1);
            layer_forward<<<grid_size, num_in>>>(curr_layer, x, out, products);
            cudaDeviceSynchronize();

            // Update the input for the next layer
            num_in = curr_layer->num_out;
            x = out;
        }
        // Return the final output values
        return x;
    }

    void free_neuron(Neuron *neuron)
    {
        for (int i = 0; i < neuron->num_in; i++)
        {
            free_value(neuron->w[i]);
        }
        cudaFree(neuron->w);
        free_value(neuron->b);
        cudaFree(neuron);
    }

    void free_layer(Layer *layer)
    {
        for (int i = 0; i < layer->num_out; i++)
        {
            free_neuron(layer->neurons[i]);
        }
        cudaFree(layer->neurons);
        cudaFree(layer);
    }

    void free_mlp(MLP *mlp)
    {
        for (int i = 0; i < mlp->num_layers; i++)
        {
            free_layer(mlp->layers[i]);
        }
        cudaFree(mlp->layers);
        cudaFree(mlp);
    }
}