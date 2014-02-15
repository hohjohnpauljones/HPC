#include <iostream>
#include <chrono>

// Recursive fibonacci sequence, this should take a while.
 uint64_t recursive_fib_seq(int x) {
     if (x == 0 || x == 1) return x;
         return recursive_fib_seq(x-1) + recursive_fib_seq(x-2);
         }
//
         int main (int argc, char *argv[]) {
//             
//                 // gets the current time (starting time).  Do this before your code that you want to time.
                     std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
//                         
//                             // Code that you want to time.
                                 std::cout << "Getting the fib sequence for of 40\n";
                                     std::cout << "Result was: " << recursive_fib_seq(40) << std::endl;
//                                         // End of the code that you want to time.
//                                             
//                                                 // Gets the new current time (ending time).  Do this after the code you want to time.
                                                     std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
//                                                         // calculates the difference in time in between start and stop.  duration<double> calculates time in seconds.
                                                             std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
//                                                                 
//                                                                     // .count returns the time in seconds.
                                                                         std::cout << "It took me " << time_span.count() << " seconds." << std::endl;
                                                                             
                                                                                 return 0;
                                                                                 }
