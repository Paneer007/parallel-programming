#include <stdio.h>
#include <stdlib.h>

#include "./cuda_grad/engine.hpp"
#include "./cuda_grad/nn.hpp"

void backprop_normal()
{
  auto x = init_value(-2);
  auto y = init_value(5);
  auto z = init_value(-4);

  auto q = x->add(y);
  auto f = q->mul(z);

  f->backward();

  f->print_expression();
  x->print_value();
  y->print_value();
  z->print_value();
}

void backprop_mlp()
{
  int n_inputs = 2;
  int n_outputs = 2;

  int sizes[] = {n_inputs, 5, 10, 5, n_outputs};
  int nlayers = sizeof(sizes) / sizeof(int);

  auto mlp = init_mlp(sizes, nlayers);

  Value **in;
  allocValueArr(&in, n_inputs);
  // Set inputs
  in[0] = init_value(1.0);
  in[1] = init_value(1.0);

  Value **out;
  allocValueArr(&out, n_inputs);
  // Set inputs
  in[0] = init_value(2);
  in[1] = init_value(3);

  // Compute outputs with forward pass
  auto out = mlp->mlp_forward(in, n_inputs);

  out[0]->print_value();
  out[1]->print_value();
}

int main(int argc, char **argv)
{

  backprop_normal();
  backprop_mlp();

  return EXIT_SUCCESS;
}