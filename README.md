# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Parallel Check


```console
(.venv) PS D:\Grad\Fall 2024\MLE\mod3-divyb> python project/parallel_check.py 
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (176)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (176)
------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                       |
        out: Storage,                                                               |
        out_shape: Shape,                                                           |
        out_strides: Strides,                                                       |
        in_storage: Storage,                                                        |
        in_shape: Shape,                                                            |
        in_strides: Strides,                                                        |
    ) -> None:                                                                      |
                                                                                    |
                                                                                    |
                                                                                    |
        # TODO: Implement for Task 3.1.                                             |
        #raise NotImplementedError("Need to implement for Task 3.1")                |
        #Raising error if in_shape is not smaller than out_shape                    |
        # Numba does not support formatted strings or f-strings                     |
                                                                                    |
        stride_aligned = np.array_equal(out_shape, in_shape) and np.array_equal(    |
            out_strides, in_strides                                                 |
        )                                                                           |
                                                                                    |
                                                                                    |
                                                                                    |
        if stride_aligned:                                                          |
            for i in prange(out.size):----------------------------------------------| #2
                out[i] = fn(in_storage[i])                                          |
                                                                                    |
        else:                                                                       |
            for i in prange(out.size):----------------------------------------------| #3
                in_index = np.zeros(len(in_shape), dtype=np.int32)------------------| #0
                out_index = np.zeros(len(out_shape), dtype=np.int32)----------------| #1
                to_index(i, out_shape, out_index)                                   |
                broadcast_index(out_index, out_shape, in_shape, in_index)           |
                in_pos = index_to_position(in_index, in_strides)                    |
                out_pos = index_to_position(out_index, out_strides)                 |
                out[out_pos] = fn(in_storage[in_pos])                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Grad\Fall
2024\MLE\mod3-divyb\minitorch\fast_ops.py (204) is hoisted out of the parallel
loop labelled #3 (it will be performed before the loop is executed and reused
inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Grad\Fall
2024\MLE\mod3-divyb\minitorch\fast_ops.py (205) is hoisted out of the parallel
loop labelled #3 (it will be performed before the loop is executed and reused
inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (238)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (238)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # TODO: Implement for Task 3.1.                                    |
        stride_aligned = (                                                 |
                                                                           |
            np.array_equal(out_shape, a_shape)                             |
            and np.array_equal(out_shape, b_shape)                         |
            and np.array_equal(out_strides, a_strides)                     |
            and np.array_equal(out_strides, b_strides)                     |
                                                                           |
        )                                                                  |
                                                                           |
                                                                           |
                                                                           |
        if stride_aligned:                                                 |
            for i in prange(out.size):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                    |
                                                                           |
        else:                                                              |
            for i in prange(out.size):-------------------------------------| #8
                                                                           |
                a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #4
                b_index = np.zeros(len(b_shape), dtype=np.int32)-----------| #5
                out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #6
                to_index(i, out_shape, out_index)                          |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                out_pos = index_to_position(out_index, out_strides)        |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4, #5, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial)
   +--6 (serial)



Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Grad\Fall
2024\MLE\mod3-divyb\minitorch\fast_ops.py (268) is hoisted out of the parallel
loop labelled #8 (it will be performed before the loop is executed and reused
inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Grad\Fall
2024\MLE\mod3-divyb\minitorch\fast_ops.py (269) is hoisted out of the parallel
loop labelled #8 (it will be performed before the loop is executed and reused
inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Grad\Fall
2024\MLE\mod3-divyb\minitorch\fast_ops.py (270) is hoisted out of the parallel
loop labelled #8 (it will be performed before the loop is executed and reused
inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (306)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (306)
------------------------------------------------------------------------|loop #ID
    def _reduce(                                                        |
        out: Storage,                                                   |
        out_shape: Shape,                                               |
        out_strides: Strides,                                           |
        a_storage: Storage,                                             |
        a_shape: Shape,                                                 |
        a_strides: Strides,                                             |
        reduce_dim: int,                                                |
    ) -> None:                                                          |
        # TODO: Implement for Task 3.1.                                 |
        #raise NotImplementedError("Need to implement for Task 3.1")    |
        reduce_size = a_shape[reduce_dim]                               |
                                                                        |
        for i in prange(len(out)):--------------------------------------| #10
            out_index = np.zeros(len(out_shape), dtype=np.int32)--------| #9
            to_index(i, out_shape, out_index)                           |
            o = index_to_position(out_index, out_strides)               |
                                                                        |
            acc = out[o]                                                |
                                                                        |
            for s in range(reduce_size):                                |
                out_index[reduce_dim] = s                               |
                j = index_to_position(out_index, a_strides)             |
                acc = fn(acc, a_storage[j])                             |
                                                                        |
            out[o] = acc                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Grad\Fall
2024\MLE\mod3-divyb\minitorch\fast_ops.py (320) is hoisted out of the parallel
loop labelled #10 (it will be performed before the loop is executed and reused
inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (336)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, D:\Grad\Fall 2024\MLE\mod3-divyb\minitorch\fast_ops.py (336)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
                                                                                          |
    for i in prange(a_shape[0]):----------------------------------------------------------| #11
        for j in range(a_shape[1]):                                                       |
            for k in range(b_shape[2]):                                                   |
                sum = 0.0                                                                 |
                                                                                          |
                for l in range(a_shape[-1]):                                              |
                    a_pos = i * a_batch_stride + j * a_strides[1] + l * a_strides[2]      |
                    b_pos = i * b_batch_stride + l * b_strides[1] + k * b_strides[2]      |
                    sum += a_storage[a_pos] * b_storage[b_pos]                            |
                                                                                          |
                out_pos = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]    |
                out[out_pos] = sum                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Simple dataset

## Cpu logs

```bash

!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

```console
Epoch  0  loss  4.933040226991119 correct 50
Epoch  10  loss  1.9982746949562835 correct 50
Epoch  20  loss  0.6338374985617523 correct 50
Epoch  30  loss  0.8976876310471699 correct 50
Epoch  40  loss  0.1752955320443651 correct 50
Epoch  50  loss  0.20297117904958709 correct 50
Epoch  60  loss  0.2020444716877168 correct 50
Epoch  70  loss  0.4731014035505514 correct 50
Epoch  80  loss  0.23061230256848753 correct 50
Epoch  90  loss  0.25328737077586694 correct 50
Epoch  100  loss  0.09871825625861631 correct 50
Epoch  110  loss  0.41612916948406337 correct 50
Epoch  120  loss  0.07394668416232371 correct 50
Epoch  130  loss  0.2611420614109699 correct 50
Epoch  140  loss  0.11033742262912333 correct 50
Epoch  150  loss  0.10243491369191107 correct 50
Epoch  160  loss  0.16093546252792354 correct 50
Epoch  170  loss  0.13198538784709202 correct 50
Epoch  180  loss  0.04776316568733375 correct 50
Epoch  190  loss  0.08996266991334267 correct 50
Epoch  200  loss  0.036109706168538314 correct 50
Epoch  210  loss  0.08005293322627217 correct 50
Epoch  220  loss  0.005178739440080159 correct 50
Epoch  230  loss  0.10612441171843211 correct 50
Epoch  240  loss  0.16593679891953386 correct 50
Epoch  250  loss  0.09146619382876302 correct 50
Epoch  260  loss  0.03904012296417532 correct 50
Epoch  270  loss  0.004718876500147546 correct 50
Epoch  280  loss  0.09337297226388498 correct 50
Epoch  290  loss  0.031091940608299475 correct 50
Epoch  300  loss  0.1674439225943422 correct 50
Epoch  310  loss  0.11235667763803307 correct 50
Epoch  320  loss  0.11860027270394013 correct 50
Epoch  330  loss  0.049225788150448 correct 50
Epoch  340  loss  0.07644045155355149 correct 50
Epoch  350  loss  0.042910821772302946 correct 50
Epoch  360  loss  0.23096540718036135 correct 50
Epoch  370  loss  0.01992044995673809 correct 50
Epoch  380  loss  0.14587000226706748 correct 50
Epoch  390  loss  0.10099122896967407 correct 50
Epoch  400  loss  0.04892607498731361 correct 50
Epoch  410  loss  0.03016758601731793 correct 50
Epoch  420  loss  0.0003621448302532139 correct 50
Epoch  430  loss  0.07445277935017576 correct 50
Epoch  440  loss  0.10369158331528375 correct 50
Epoch  450  loss  0.039862280436325885 correct 50
Epoch  460  loss  0.09019894195436795 correct 50
Epoch  470  loss  0.01984565996004067 correct 50
Epoch  480  loss  0.07724024891922297 correct 50
Epoch  490  loss  0.003096335619505459 correct 50
Average epoch time: 0.1533s
```

## gpu logs

```bash

!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

```console
Epoch  0  loss  6.928547857429484 correct 26
Epoch  10  loss  6.50788713605911 correct 26
Epoch  20  loss  6.164391331506948 correct 36
Epoch  30  loss  6.207926981107077 correct 37
Epoch  40  loss  5.8532983652964905 correct 38
Epoch  50  loss  5.048503891107041 correct 39
Epoch  60  loss  5.026963010381639 correct 41
Epoch  70  loss  4.289923894772958 correct 47
Epoch  80  loss  3.0981782605679884 correct 47
Epoch  90  loss  2.8963747756960214 correct 50
Epoch  100  loss  1.7396440398192188 correct 48
Epoch  110  loss  0.7518608840282578 correct 43
Epoch  120  loss  2.1637682771407185 correct 49
Epoch  130  loss  0.5256122317022883 correct 48
Epoch  140  loss  0.49788757253486227 correct 50
Epoch  150  loss  1.4923934408338646 correct 49
Epoch  160  loss  0.15695845926158072 correct 48
Epoch  170  loss  0.8186240984245036 correct 48
Epoch  180  loss  0.1868169660133045 correct 49
Epoch  190  loss  0.15438066292016395 correct 48
Epoch  200  loss  1.2129487299119164 correct 46
Epoch  210  loss  0.9334486686954364 correct 48
Epoch  220  loss  1.3220342660757767 correct 48
Epoch  230  loss  0.052101148608116425 correct 50
Epoch  240  loss  0.08093526425290586 correct 50
Epoch  250  loss  1.9971267664079637 correct 49
Epoch  260  loss  0.5866672428017312 correct 49
Epoch  270  loss  0.00969228109681768 correct 49
Epoch  280  loss  1.4608641953537769 correct 48
Epoch  290  loss  0.06067043566952632 correct 49
Epoch  300  loss  0.19429067409701223 correct 50
Epoch  310  loss  0.3026770004353011 correct 50
Epoch  320  loss  0.010754390417781553 correct 49
Epoch  330  loss  0.01904760822564454 correct 50
Epoch  340  loss  0.17246032874193024 correct 49
Epoch  350  loss  0.4230174926276938 correct 50
Epoch  360  loss  0.0848288654927252 correct 49
Epoch  370  loss  0.2253847807625591 correct 50
Epoch  380  loss  0.010626362674940184 correct 50
Epoch  390  loss  0.04098812231161347 correct 50
Epoch  400  loss  0.1146244770429517 correct 50
Epoch  410  loss  0.003771276629054045 correct 49
Epoch  420  loss  1.5650914869344894 correct 49
Epoch  430  loss  0.2000754627035671 correct 50
Epoch  440  loss  0.03617538804424886 correct 50
Epoch  450  loss  0.01023820813650585 correct 50
Epoch  460  loss  0.012271378145512603 correct 49
Epoch  470  loss  0.4685617982989177 correct 50
Epoch  480  loss  0.644554319232363 correct 48
Epoch  490  loss  0.04367905913934092 correct 49
Average epoch time: 1.6107s
```

# Split dataset

## Cpu logs

```bash

!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

```console
Epoch  0  loss  7.219599348876483 correct 29
Epoch  10  loss  4.560914294638067 correct 39
Epoch  20  loss  4.566004765420733 correct 40
Epoch  30  loss  3.59083557438109 correct 43
Epoch  40  loss  4.093013344154103 correct 44
Epoch  50  loss  3.1798243338275802 correct 44
Epoch  60  loss  3.8435958848452496 correct 46
Epoch  70  loss  2.6496782345392047 correct 44
Epoch  80  loss  3.26337392562177 correct 49
Epoch  90  loss  3.2103998707019183 correct 42
Epoch  100  loss  0.990144904972022 correct 43
Epoch  110  loss  1.3487287261052923 correct 48
Epoch  120  loss  1.9990548008585605 correct 46
Epoch  130  loss  1.4319740208001661 correct 48
Epoch  140  loss  3.4687405337559585 correct 48
Epoch  150  loss  1.5625337109394977 correct 48
Epoch  160  loss  0.4807743391672621 correct 48
Epoch  170  loss  2.5349842101573383 correct 48
Epoch  180  loss  1.3538398059943162 correct 49
Epoch  190  loss  1.6414360008505127 correct 48
Epoch  200  loss  1.6573956494724176 correct 49
Epoch  210  loss  2.505017196237248 correct 48
Epoch  220  loss  0.8676324076140184 correct 47
Epoch  230  loss  1.7283059067084465 correct 49
Epoch  240  loss  1.5283782772951946 correct 49
Epoch  250  loss  0.6698726196599176 correct 48
Epoch  260  loss  1.9111731525324696 correct 49
Epoch  270  loss  1.2110599940166022 correct 48
Epoch  280  loss  1.623117901069838 correct 48
Epoch  290  loss  1.0069683816446613 correct 48
Epoch  300  loss  0.425854965825398 correct 49
Epoch  310  loss  1.07631474552386 correct 48
Epoch  320  loss  1.193796866985044 correct 49
Epoch  330  loss  0.3771979641471653 correct 48
Epoch  340  loss  1.3827635471353386 correct 49
Epoch  350  loss  0.5521062350570188 correct 50
Epoch  360  loss  0.8198859028625803 correct 48
Epoch  370  loss  0.42600539081814226 correct 49
Epoch  380  loss  1.3763115760173297 correct 49
Epoch  390  loss  1.2776893173375374 correct 50
Epoch  400  loss  0.3274831715434621 correct 48
Epoch  410  loss  1.81216705731846 correct 48
Epoch  420  loss  2.153297857908585 correct 49
Epoch  430  loss  1.1101168300271669 correct 49
Epoch  440  loss  0.560884775976242 correct 49
Epoch  450  loss  1.3865431620445405 correct 49
Epoch  460  loss  2.137669204742669 correct 47
Epoch  470  loss  1.7183916650753017 correct 49
Epoch  480  loss  0.11301081447802284 correct 49
Epoch  490  loss  1.3326465833110248 correct 50
Average epoch time: 0.1532s
```

## gpu logs

```bash

!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

```console
Epoch  0  loss  7.661211970020025 correct 10
Epoch  10  loss  8.864347690970018 correct 10
Epoch  20  loss  7.715963017105352 correct 8
Epoch  30  loss  3.66027875407032 correct 7
Epoch  40  loss  4.96063837600802 correct 7
Epoch  50  loss  7.820741213968494 correct 8
Epoch  60  loss  6.345438221643725 correct 9
Epoch  70  loss  5.130693472615098 correct 15
Epoch  80  loss  3.637715436238656 correct 17
Epoch  90  loss  5.050534778216473 correct 18
Epoch  100  loss  4.915959427034697 correct 16
Epoch  110  loss  7.84182271825042 correct 23
Epoch  120  loss  3.5488699412609352 correct 36
Epoch  130  loss  6.264888526038361 correct 41
Epoch  140  loss  2.5833820486397343 correct 25
Epoch  150  loss  5.972992137064231 correct 11
Epoch  160  loss  4.864752113413149 correct 40
Epoch  170  loss  3.707631468924829 correct 40
Epoch  180  loss  4.717759305570901 correct 41
Epoch  190  loss  6.2023125061977495 correct 41
Epoch  200  loss  5.7771118500759595 correct 42
Epoch  210  loss  7.742355226770194 correct 40
Epoch  220  loss  5.3017348225773775 correct 40
Epoch  230  loss  8.86453822808302 correct 43
Epoch  240  loss  5.542363687660364 correct 23
Epoch  250  loss  6.673458274577026 correct 40
Epoch  260  loss  2.8939852906904564 correct 37
Epoch  270  loss  4.020901349308758 correct 42
Epoch  280  loss  5.955098350769386 correct 44
Epoch  290  loss  5.41073063858386 correct 41
Epoch  300  loss  3.313166399863014 correct 40
Epoch  310  loss  3.521922282564647 correct 36
Epoch  320  loss  4.306257088862846 correct 25
Epoch  330  loss  3.101548101732695 correct 20
Epoch  340  loss  3.469803910481846 correct 37
Epoch  350  loss  3.0376342459649157 correct 45
Epoch  360  loss  1.115044980021005 correct 43
Epoch  370  loss  1.2708784399219661 correct 44
Epoch  380  loss  1.6360039027432443 correct 41
Epoch  390  loss  6.519852401276545 correct 41
Epoch  400  loss  1.5637913369361764 correct 45
Epoch  410  loss  0.8244315956073021 correct 47
Epoch  420  loss  1.037787581283316 correct 46
Epoch  430  loss  1.0430675873528918 correct 49
Epoch  440  loss  2.1800589282800398 correct 49
Epoch  450  loss  6.679288107047853 correct 50
Epoch  460  loss  6.793712091878893 correct 50
Epoch  470  loss  2.3000100239516725 correct 50
Epoch  480  loss  0.42497491301520435 correct 50
Epoch  490  loss  3.507153997201661 correct 50
Average epoch time: 1.6275s

```
# Split dataset

## Cpu logs

```bash

!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

```console
Epoch  0  loss  6.360264403185886 correct 24
Epoch  10  loss  5.577388060290142 correct 40
Epoch  20  loss  4.338418522874343 correct 45
Epoch  30  loss  2.9255356501354073 correct 41
Epoch  40  loss  3.2724342670819166 correct 42
Epoch  50  loss  3.365018316652416 correct 43
Epoch  60  loss  3.3768230237500614 correct 45
Epoch  70  loss  2.2947875319273816 correct 43
Epoch  80  loss  1.628693169759706 correct 43
Epoch  90  loss  2.3802507362608982 correct 46
Epoch  100  loss  3.0736410500946247 correct 45
Epoch  110  loss  1.2709024672308642 correct 46
Epoch  120  loss  1.723909995114108 correct 43
Epoch  130  loss  1.8129679524044822 correct 46
Epoch  140  loss  3.6339446098231716 correct 44
Epoch  150  loss  3.0549589322010586 correct 46
Epoch  160  loss  2.783511290462406 correct 47
Epoch  170  loss  3.684056496656758 correct 45
Epoch  180  loss  2.9345300186010848 correct 47
Epoch  190  loss  0.8837784063760986 correct 47
Epoch  200  loss  2.4707862941446916 correct 48
Epoch  210  loss  3.1308044540934565 correct 46
Epoch  220  loss  1.4810251292222756 correct 47
Epoch  230  loss  3.885914932388605 correct 43
Epoch  240  loss  0.503588382910265 correct 48
Epoch  250  loss  1.0283429446554202 correct 46
Epoch  260  loss  2.133789531657713 correct 46
Epoch  270  loss  0.3910393147335152 correct 48
Epoch  280  loss  1.0390435404630292 correct 48
Epoch  290  loss  2.226081339582859 correct 49
Epoch  300  loss  1.2914187329251805 correct 50
Epoch  310  loss  0.7396662709106575 correct 47
Epoch  320  loss  1.3010727629526126 correct 50
Epoch  330  loss  1.0999414372043022 correct  0
Epoch  340  loss  2.341594861267124 correct 47
Epoch  350  loss  1.9546834760354226 correct 49
Epoch  360  loss  0.17052654254256194 correct 50
Epoch  370  loss  0.2861595917199935 correct 49
Epoch  380  loss  0.9514257093059935 correct 50
Epoch  390  loss  2.0074455877557664 correct 49
Epoch  400  loss  0.7230514640058165 correct 50
Epoch  410  loss  1.0730296173679181 correct 50
Epoch  420  loss  0.9749325150886705 correct 50
Epoch  430  loss  1.425602756194041 correct 50
Epoch  440  loss  0.289665241995709 correct 49
Epoch  450  loss  0.9586566409452465 correct 50
Epoch  460  loss  1.7194460076697515 correct 48
Epoch  470  loss  0.9678140546098547 correct 50
Epoch  480  loss  0.6721902742441828 correct 49
Epoch  490  loss  0.3936038941333855 correct 50
Average epoch time: 0.1524s
```

## gpu logs

```bash

!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpy --HIDDEN 100 --DATASET xor --RATE 0.05
```

```console
