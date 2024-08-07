#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

using namespace std;

// Shared variables
volatile bool flag[2] = {false, false};
volatile int victim = 0;

// Lock function based on Peterson's solution
void lock(int me) {
  int other = 1 - me; // The other thread index
  flag[me] = true;
  victim = me;
  while (flag[other] && victim == me) {
    this_thread::sleep_for(chrono::milliseconds(1));
  }
}

void unlock(int me) { flag[me] = false; }

// Critical section function
void critical_section(int id) {
  lock(id);
  cout << "Thread " << id << " is in the critical section." << endl;
  this_thread::sleep_for(chrono::milliseconds(100));
  unlock(id);
}

int main() {
  thread t1(critical_section, 0);
  thread t2(critical_section, 1);

  t1.join();
  t2.join();

  return 0;
}
