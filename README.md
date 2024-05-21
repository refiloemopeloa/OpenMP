# OpenMP

Learning and exploring the world of shared memory parallelism through OpenMP

# Using OpenMP

## Import library

> `#include "omp.h"`

## Directives

> `#pragma omp parallel`

give me the default number of threads and creates a parallel region

---

> `omp_get_thread_num()`

get the ID number of the current thread

---

> `omp_set_num_threads(n)`

sets n number of threads for a parallel region

---

> `omp_get_wtime()`

returns the time at the point when the function was called

---

> `#pragma omp barrier`

sets a point in the parallel block where every thread will stop until all threads have reached the barrier

---

> `#pragma omp critical`

only one thread at a time will execute the code in the critical section

---

> `#pragma omp atomic`

lightweight critical

---

## omp for

> `#pragma omp for`

split up iterations of a for loop among threads

---

> `#pragma omp for schedule`

how iterations of the loop are broken up

-   > `schedule(static [,chunk])`

    -   deal out iterations in the loop in a round robin fashion

    -   > `chunk`

        -   by default, chunk is set to 1

-   > `schedule(dynamic [,chunk])`

    -   put loop iterations into a task queue

    -   threads will take iterations from queue and run them
    -   threads will only fetch another iteration once previous iteration is done
    -   occurs during run-time

-   > `schedule(guided [,chunk])`
    -   start dynamic
    -   start with a large chunk size
    -   progressively shrink chunk size
-   > `schedule(runtime)`
    -   pass schedule and chunk size at runtime
-   > `schedule(auto)`
    -   schedule is left to compiler
-   > `omp_set_schedule(schedule)`
    -   set the type of schedule
-   > `omp_get_schedule()`
    -   get the schedule type

> `#pragma omp for reduction(operation: list)`

-   create a local copy of the variable in `list` that is local to each thread
-   combines the local copies into a single global copy using operator specified in `operation`
-   combines that with the global copy

> `#pragma omp for nowait`

-   skip barrier at the end of this loop

---

> `#pragma omp master`

-   only the master thread will do this code

---

> `#pragma omp single`

-   only one thread will do the work in the block
-   the first thread that gets to it
-   if you aren't the thread that got it, you wait at the end of the block
-   can add nowait
    > `#pragma omp sections`
-   tells the compiler to expect sections
    > `#pragma omp section`
-   create a section of code that one thread will do
    > `omp_init_lock()`
-   initialise lock variable

---

> `omp_set_lock()`

-   thread grabs lock

> `omp_unset_lock()`

-   allows other threads to grab the lock
    > `omp_destroy_lock()`
-   frees the lock
    > `omp_test_lock()`
-   checks if the lock is available
    > `omp_get_max_threads()`
-   get max allocated threads
    > `omp_in_parallel()`
-   check if in parallel region

---

> `omp_set_dynamic(n)`

-   set dynamic mode
-   `n=0 ` turns off dynamic mode
    > `omp_get_dynamic()`
-   check if in dynamic mode
    > `omp_num_procs()`
-   how many processors at runtime
    > `OMP_NUM_THREADS`
-   by default, how many threads to use

---

> `OMP_STACKSIZE`

-   how much stack the system should set aside
    > `OMP_WAIT_POLICY`
-   what to do with barriers
-   > `ACTIVE | PASSIVE`
        - >`ACTIVE`
            - thread spins, waiting for a lock
        - >`PASSIVE`
            -put thread to sleep

---

> `OMP_PROC_BIND`

-   bind a thread to a processor
-   > `TRUE | FALSE` >`#pragma omp parallel shared(list)`
-   variables in list are shared among the threads

---

> `#pragma omp parallel private(list)`

-   creates private instance of variables in list for each thread
-   variables are uninitialized

> `#pragma omp parallel firstprivate(list)`

-   private but initializes variable to global value

`#pragma omp parallel lastprivate()`

-   the last value seen at the last iteration is the value that the variable is set to
-   which ever thread does the last iteration of the loop sends that value to the global variable

---

> `#pragma omp parallel default(clause)`

-   clause:
    -   > `none`
        -   compiler requires explicit definition of data attribute of each variable
    -   > `shared`
        -   by default
        -   declares all variables shared
    -   > `private`
        -   not available in c/c++

---

> `#pragma omp task`

-   variables in task are firstprivate by default

---

> `#pragma omp threadprivate(list)`

-   variable is global to the scope but private per thread

---

> `#pragma omp parallel copyin(list)`

-   each thread will see the value of the variables in list
