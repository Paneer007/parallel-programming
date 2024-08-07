#include "engine.hpp"

// Node in a neural network
class Neuron
{
public:
  Value **w;  // array of weights
  Value *b;   // bias
  int num_in; // number of input neurons
};

// One layer of Neurons in a neural network
class Layer
{
public:
  Neuron **neurons; // array of neurons
  int num_out;      // number of output neurons
};

// All the Corresponding layers in a neural network
class MLP
{
public:
  Layer **layers; // array of layers
  int num_layers; // number of layers

  // Forward Propagation of values in a neural network
  Value **mlp_forward(Value **x, int nin);

  // TODO: Implement Back propagation
  // TODO: Implement Updation of weights
};

// Init a MLP using unified memory
MLP *init_mlp(int *sizes, int nlayers);

// Free memory
void free_neuron(Neuron *neuron);
void free_layer(Layer *layer);
void free_mlp(MLP *mlp);