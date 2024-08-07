
#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <stdlib.h>

// List of operators supported by the backpropagation engine
// Special operator to indicate absence of an operator
typedef enum Operator
{
  NUL,
  ADD,
  SUB,
  MUL,
  DIV,
  POW
} Operator;

// Value is used to represent the data
// It comprises as the basic unit of the computational graph
// The computational graph is used to compute the gradients in backpropagation
class Value
{
public:
  float val;        // value of node in computational graph
  float grad;       // gradient of node in computational graph
  Value **children; // children of the node in the computational graph
  int n_children;   // number of children

  Operator op; // Operation to be performed on the node

  // TODO: Fix the operator overloading

  Value *operator+(Value *b);
  Value *add(Value *b);

  Value *operator-(Value *b);
  Value *sub(Value *b);

  Value *operator*(Value *b);
  Value *mul(Value *b);

  Value *operator/(Value *b);
  Value *divide(Value *b);

  Value *operator^(Value *b);
  Value *power(Value *b);

  // Utility functions to display the graph
  void print_value();
  void print_children();
  void print_expression();

  // Back propagation utility functions
  void build_topo(Value **topo, int *topo_size, Value **visited,
                  int *visited_size);
  void backward();
};

// Functions to create an instance of the Value class
// Why do we not use the constructor?
// We use cudaMallocManaged to utilise the unified memory rather than manually
// manage memory
Value *init_value(float x);
Value **init_values(float *arr, size_t len);

// Utility functions to allocate unified memory b/w CPU and GPU
void allocValue(Value **v, size_t num);
void allocValueArr(Value ***ptr, size_t len);

// Free memory
void free_value(Value *v);

#endif