#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

const int N = 10;     // Number of threads (change as needed)
atomic<bool> flag[N]; // Indicates whether a thread is interested in
                      // entering the critical section
atomic<int> label[N]; // The "label" or priority of each thread

void lock(int me) {
  int maxLabel = 0;
  flag[me] = true;
  label[me] = 1 + *max_element(label, label + N);

  for (int k = 0; k < N; ++k) {
    if (k != me) {
      while (flag[k] &&
             (label[k] < label[me] || (label[k] == label[me] && k < me))) {
        this_thread::sleep_for(
            chrono::milliseconds(1)); // Prevent tight looping
      }
    }
  }
}

void unlock(int me) {
  flag[me] =
      false; // Indicate that thread `me` is done with the critical section
}

void critical_section(int id) {
  lock(id);
  // Critical section begins
  cout << "Thread " << id << " is in the critical section." << endl;
  this_thread::sleep_for(
      chrono::milliseconds(100)); // Simulate work in the critical section
  // Critical section ends
  unlock(id);
}

int main() {
  vector<thread> threads;

  // Initialize flag and label arrays
  for (int i = 0; i < N; ++i) {
    flag[i] = false;
    label[i] = 0;
  }

  // Create threads
  for (int i = 0; i < N; ++i) {
    threads.emplace_back(critical_section, i);
  }

  // Join threads
  for (auto &t : threads) {
    t.join();
  }

  return 0;
}