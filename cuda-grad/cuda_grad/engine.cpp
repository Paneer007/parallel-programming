#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "engine.hpp"

// Initialises the Value Object
Value *init_value(float x)
{
  Value *v;
  allocValue(&v, 1);
  v->val = x;
  v->grad = 0;
  v->children = NULL;
  v->n_children = 0;
  v->op = NUL;
  return v;
}

// Initialises an Array of values with a given array
Value **init_values(float *arr, size_t len)
{
  Value **values;
  allocValueArr(&values, len);
  if (values == NULL)
  {
    perror("Memory allocation for values failed");
    exit(1);
  }
  for (size_t i = 0; i < len; i++)
  {
    values[i] = init_value(arr[i]);
  }
  return values;
}

// Returns a new node as a result of an operator
Value *binary_op_value(float val, Value *a, Value *b, Operator op)
{
  Value *out;
  allocValue(&out, 1);
  out->val = val;
  out->grad = 0;
  // Allocate memory for children
  allocValueArr(&(out->children), 2);
  // Set children to pointers of a and b
  out->children[0] = a;
  out->children[1] = b;
  out->n_children = 2;
  out->op = op;
  return out;
}

// Addition Operator
Value *add_values(Value *a, Value *b)
{
  return binary_op_value(a->val + b->val, a, b, ADD);
}
Value *Value::add(Value *b) { return add_values(this, b); }
Value *Value::operator+(Value *b) { return add_values(this, b); }

// Subtraction Operator
Value *sub_values(Value *a, Value *b)
{
  return binary_op_value(a->val - b->val, a, b, SUB);
}
Value *Value::sub(Value *b) { return sub_values(this, b); }
Value *Value::operator-(Value *b) { return sub_values(this, b); }

// Multiplication Operator
Value *mul_values(Value *a, Value *b)
{
  return binary_op_value(a->val * b->val, a, b, MUL);
}
Value *Value::mul(Value *b) { return mul_values(this, b); }
Value *Value::operator*(Value *b) { return mul_values(this, b); }

// Division Operator
Value *divide_values(Value *a, Value *b)
{
  if (b->val == 0.0)
  {
    printf("Error: Division by zero\n");
    exit(1);
  }
  return binary_op_value(a->val / b->val, a, b, DIV);
}
Value *Value::divide(Value *b) { return divide_values(this, b); }
Value *Value::operator/(Value *b) { return divide_values(this, b); }

// Exponential Operator
Value *power_values(Value *a, Value *b)
{
  return binary_op_value(pow(a->val, b->val), a, b, POW);
}
Value *Value::power(Value *b) { return power_values(this, b); }
Value *Value::operator^(Value *b) { return power_values(this, b); }

// Constructs the topological graph
void Value::build_topo(Value **topo, int *topo_size, Value **visited,
                       int *visited_size)
{

  // Iterates through the visited nodes and checks if current node is processed
  // or not
  for (int i = 0; i < *visited_size; ++i)
  {
    if (visited[i] == this)
      return;
  }

  // Appends new node to the set of visited nodes
  visited[*visited_size] = this;
  (*visited_size)++;

  // Recursively traverses the children nodes
  for (int i = 0; i < this->n_children; ++i)
  {
    this->children[i]->build_topo(topo, topo_size, visited, visited_size);
  }

  // Appends the node to the end of the topological order
  topo[*topo_size] = this;
  (*topo_size)++;
}

void Value::print_value()
{
  printf("Value(val=%.2f, grad=%.2f)\n", this->val, this->grad);
}

// Recursively prints the children of a node and the value/operator associated
// with a value
void Value::print_children()
{
  for (int i = 0; i < this->n_children; i++)
  {
    this->children[i]->print_children();
  }

  char operand;
  switch (this->op)
  {
  case ADD:
    operand = '+';
    break;
  case SUB:
    operand = '-';
    break;
  case MUL:
    operand = '*';
    break;
  case DIV:
    operand = '/';
    break;
  case POW:
    operand = '^';
    break;
  default:
    operand = ' ';
    break;
  }

  if (this->n_children == 0)
  {
    printf("%.2f ", this->val);
  }
  else
  {
    printf("%c ", operand);
  }
}

void Value::print_expression()
{
  this->print_children();
  printf("= %.2f\n", this->val);
}
