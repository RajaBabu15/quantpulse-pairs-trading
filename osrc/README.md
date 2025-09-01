# QuantPulse Pairs Trading - OCaml Implementation

This directory contains the OCaml equivalent implementations of the C++ source files from the `csrc` directory. The OCaml version provides the same functionality as the C++ implementation but leverages OCaml's functional programming paradigms, strong type system, and memory safety.

## File Mappings

| C++ File | OCaml File | Description |
|----------|------------|-------------|
| `trading_engine.h` | `trading_engine.mli` | Interface definitions and type declarations |
| `trading_engine.h` | `trading_engine.ml` | Core trading engine implementation |
| `vectorized_math.cpp` | `vectorized_math.ml` | Mathematical operations and SIMD-style vectorization |
| `cross_validation.cpp` | `cross_validation.ml` | Cross-validation and parallel optimization |
| `performance_cache.cpp` | `performance_cache.ml` | LRU cache implementation for performance optimization |
| `python_bindings.cpp` | `ocaml_bindings.ml` | Native OCaml bindings and interface |

## Key Features

### 1. **Type Safety**
- Strong static typing prevents runtime errors
- Algebraic data types for structured data representation
- Pattern matching for exhaustive case handling

### 2. **Functional Programming**
- Immutable data structures by default
- Higher-order functions for data processing
- Pure functions without side effects where possible

### 3. **Memory Safety**
- Automatic garbage collection
- No manual memory management required
- Bounds checking for array operations

### 4. **Performance Optimization**
- Efficient array operations using OCaml's native arrays
- Functional approach to SIMD-style vectorization
- Thread-safe caching with LRU eviction

### 5. **Concurrency**
- Lightweight threading using OCaml threads
- Async programming with Lwt (Lightweight threads)
- Thread-safe data structures with mutexes

## Architecture

### Core Modules

1. **Trading_engine** - Central module containing:
   - Type definitions for trading parameters and results
   - Thread pool implementation
   - Optimization cache
   - Core trading functions

2. **Vectorized_math** - Mathematical operations:
   - SIMD-style array operations
   - Statistical functions
   - Performance utilities

3. **Cross_validation** - Optimization and validation:
   - Parallel cross-validation
   - Backtest implementation
   - Parameter optimization

4. **Performance_cache** - Caching system:
   - Thread-safe LRU cache
   - Spread calculation caching
   - Performance monitoring

5. **Ocaml_bindings** - High-level interface:
   - User-friendly API
   - Advanced analytics
   - Optimization utilities

### Key Differences from C++ Implementation

1. **Memory Management**: OCaml's garbage collector eliminates manual memory management
2. **Error Handling**: Pattern matching and exceptions instead of return codes
3. **Concurrency**: OCaml threads and Lwt instead of std::thread and OpenMP
4. **Vectorization**: Functional array operations instead of ARM NEON SIMD
5. **Data Structures**: Immutable by default with explicit mutability when needed

## Building and Usage

### Prerequisites
- OCaml 4.12+ or 5.0+
- Dune build system
- Lwt library for async operations

### Building
```bash
cd osrc
dune build
```

### Running
```bash
dune exec ./ocaml_bindings.exe
```

### Testing
```bash
# Create sample test
ocaml -I _build/default ocaml_bindings.ml
```

## Performance Characteristics

### Advantages
- **Memory Safety**: No segmentation faults or buffer overflows
- **Type Safety**: Compile-time error detection
- **Functional Programming**: Easier reasoning about code correctness
- **Garbage Collection**: Automatic memory management

### Trade-offs
- **Performance**: Slightly slower than hand-optimized C++ with SIMD
- **Memory Usage**: GC overhead compared to manual allocation
- **Real-time**: GC pauses may affect real-time requirements

## Code Examples

### Basic Usage
```ocaml
(* Load the module *)
open Ocaml_bindings;;

(* Generate sample price data *)
let prices1 = Array.init 1000 (fun i -> 
  100.0 +. 10.0 *. sin (float_of_int i /. 50.0));;

let prices2 = Array.init 1000 (fun i -> 
  95.0 +. 12.0 *. sin (float_of_int i /. 55.0));;

(* Run backtest *)
let params = [("lookback", 20.0); ("z_entry", 2.0); ("z_exit", 0.5)];;
let result = QuantPulseCore.vectorized_backtest prices1 prices2 params ();;

(* Print results *)
List.iter (fun (k, v) -> Printf.printf "%s: %.4f\n" k v) 
  (to_param_dict result);;
```

### Advanced Analytics
```ocaml
(* Calculate risk metrics *)
let trade_returns = QuantPulseCore.backtest_trade_returns prices1 prices2 params;;
let sharpe = AdvancedAnalytics.calculate_sharpe_ratio trade_returns;;
let max_dd = AdvancedAnalytics.calculate_max_drawdown equity_curve;;
let var_95 = AdvancedAnalytics.calculate_value_at_risk trade_returns 0.95;;
```

### Optimization
```ocaml
(* Generate parameter grid *)
let param_grid = OptimizationUtils.generate_parameter_grid 
  [10; 20; 30] [1.5; 2.0; 2.5] [0.3; 0.5; 0.7];;

(* Run batch optimization *)
let results = QuantPulseCore.batch_parameter_optimization 
  prices1 prices2 param_grid;;
```

## Future Enhancements

1. **Parallel Processing**: Integration with Domainslib for parallel arrays
2. **GPU Computing**: OCaml bindings to CUDA/OpenCL
3. **Streaming**: Real-time data processing with reactive streams
4. **Interop**: C bindings for performance-critical sections
5. **Distribution**: MPI bindings for distributed computing

## Comparison with C++ Version

| Aspect | C++ Version | OCaml Version |
|--------|-------------|---------------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Memory Safety | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Type Safety | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Code Maintainability | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Development Speed | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Concurrency | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

The OCaml implementation provides a more maintainable and safer alternative to the C++ version while maintaining good performance characteristics for quantitative finance applications.
