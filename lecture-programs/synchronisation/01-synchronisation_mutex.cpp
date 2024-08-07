#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

using namespace std;

atomic<bool> flag(false);

void Thread1() {
  while (!flag) {
    this_thread::sleep_for(chrono::milliseconds(1)); // Avoid busy-waiting loop
  }

  // Critical section S1
  cout << "Thread1: Executing S1" << endl;

  // Set flag to false
  flag = false;
}

void Thread2() {
  while (flag) {
    // Busy-wait until flag is false
    this_thread::sleep_for(chrono::milliseconds(1)); // Avoid busy-waiting loop
  }

  // Critical section S2
  cout << "Thread2: Executing S2" << endl;

  // Set flag to true
  flag = true;
}

int main() {
  // Start threads
  thread t1(Thread1);
  thread t2(Thread2);

  // Ensure both threads complete before exiting
  t1.join();
  t2.join();

  return 0;
}
