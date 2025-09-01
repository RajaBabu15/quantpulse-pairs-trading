(* Performance Cache - OCaml Implementation *)
(* Equivalent to performance_cache.cpp *)

open Trading_engine
open Vectorized_math
open Cross_validation

(* Thread-safe LRU Cache implementation *)
module ThreadSafeLRUCache = struct
  type ('k, 'v) cache_item = {
    key : 'k;
    mutable value : 'v;
    mutable access_count : int;
    mutable last_access : float;
  }
  
  type ('k, 'v) t = {
    items : ('k, 'v) cache_item list ref;
    cache_map : ('k, ('k, 'v) cache_item) Hashtbl.t;
    max_size : int;
    mutex : Mutex.t;
    mutable hits : int;
    mutable misses : int;
    mutable evictions : int;
  }
  
  let create max_size =
    {
      items = ref [];
      cache_map = Hashtbl.create max_size;
      max_size = max_size;
      mutex = Mutex.create ();
      hits = 0;
      misses = 0;
      evictions = 0;
    }
  
  let move_to_front cache item =
    let items_list = !(cache.items) in
    let filtered = List.filter (fun x -> x.key <> item.key) items_list in
    item.access_count <- item.access_count + 1;
    item.last_access <- Unix.time ();
    cache.items := item :: filtered
  
  let get cache key =
    Mutex.lock cache.mutex;
    (try
       let item = Hashtbl.find cache.cache_map key in
       move_to_front cache item;
       cache.hits <- cache.hits + 1;
       Mutex.unlock cache.mutex;
       Some item.value
     with Not_found ->
       cache.misses <- cache.misses + 1;
       Mutex.unlock cache.mutex;
       None)
  
  let put cache key value =
    Mutex.lock cache.mutex;
    (try
       let item = Hashtbl.find cache.cache_map key in
       item.value <- value;
       move_to_front cache item;
       Mutex.unlock cache.mutex
     with Not_found ->
       let new_item = {
         key = key;
         value = value;
         access_count = 1;
         last_access = Unix.time ();
       } in
       
       let items_list = !(cache.items) in
       cache.items := new_item :: items_list;
       Hashtbl.add cache.cache_map key new_item;
       
       if List.length !(cache.items) > cache.max_size then begin
         let sorted_items = List.sort (fun a b -> 
           compare a.last_access b.last_access
         ) !(cache.items) in
         
         match List.rev sorted_items with
         | oldest :: rest ->
             cache.items := List.rev rest;
             Hashtbl.remove cache.cache_map oldest.key;
             cache.evictions <- cache.evictions + 1
         | [] -> ()
       end;
       Mutex.unlock cache.mutex)
  
  let clear cache =
    Mutex.lock cache.mutex;
    cache.items := [];
    Hashtbl.clear cache.cache_map;
    Mutex.unlock cache.mutex
  
  let size cache =
    Mutex.lock cache.mutex;
    let len = List.length !(cache.items) in
    Mutex.unlock cache.mutex;
    len
  
  let hit_rate cache =
    let total = cache.hits + cache.misses in
    if total > 0 then (float_of_int cache.hits) /. (float_of_int total)
    else 0.0
  
  type stats = {
    hits : int;
    misses : int;
    evictions : int;
    size : int;
    hit_rate : float;
  }
  
  let get_stats cache =
    {
      hits = cache.hits;
      misses = cache.misses;
      evictions = cache.evictions;
      size = size cache;
      hit_rate = hit_rate cache;
    }
end

(* Parameter hashing *)
module ParameterHash = struct
  let hash params =
    let h = ref 0 in
    Array.iter (fun x ->
      let bits = Int64.bits_of_float x in
      h := !h lxor (Hashtbl.hash bits) + 0x9e3779b9 + (!h lsl 6) + (!h lsr 2)
    ) params;
    !h
end

(* Global cache instances *)
let backtest_cache = ThreadSafeLRUCache.create 5000
let objective_cache = ThreadSafeLRUCache.create 10000
let spread_cache = ThreadSafeLRUCache.create 1000

(* Spread cache utilities *)
let generate_spread_cache_key prices1 prices2 lookback =
  let hash1 = ref 0 in
  let hash2 = ref 0 in
  let n = Array.length prices1 in
  let step = 8 in
  
  for i = 0 to (n - 1) / step do
    let idx = i * step in
    if idx < n then begin
      hash1 := !hash1 lxor (Hashtbl.hash prices1.(idx)) + 0x9e3779b9 + (!hash1 lsl 6) + (!hash1 lsr 2);
      hash2 := !hash2 lxor (Hashtbl.hash prices2.(idx)) + 0x9e3779b9 + (!hash2 lsl 6) + (!hash2 lsr 2)
    end
  done;
  
  Printf.sprintf "spread_%d_%d_%d_%d" !hash1 !hash2 n lookback

let get_cached_spread_stats prices1 prices2 lookback =
  let key = generate_spread_cache_key prices1 prices2 lookback in
  match ThreadSafeLRUCache.get spread_cache key with
  | Some cached_data ->
      let n = Array.length prices1 in
      if Array.length cached_data = 2 * n then
        let spread = Array.sub cached_data 0 n in
        let z_scores = Array.sub cached_data n n in
        Some (spread, z_scores)
      else None
  | None -> None

let cache_spread_stats prices1 prices2 lookback spread z_scores =
  let key = generate_spread_cache_key prices1 prices2 lookback in
  let n = Array.length prices1 in
  let cache_data = Array.make (2 * n) 0.0 in
  Array.blit spread 0 cache_data 0 n;
  Array.blit z_scores 0 cache_data n n;
  ThreadSafeLRUCache.put spread_cache key cache_data

(* Cached backtest implementation *)
let cpp_cached_vectorized_backtest prices1 prices2 params =
  Printf.printf "💾 ENTERING cpp_cached_vectorized_backtest() at %s\n" (get_current_time ());
  
  let key_params = [|
    float_of_int params.lookback;
    params.z_entry;
    params.z_exit;
    float_of_int params.position_size;
    params.transaction_cost;
    params.profit_target;
    params.stop_loss;
  |] in
  
  let n = Array.length prices1 in
  let sample_size = min n 100 in
  let step = 10 in
  let price_hash = ref 0 in
  
  for i = 0 to (sample_size - 1) / step do
    let idx = i * step in
    if idx < n then begin
      let bits1 = Int64.bits_of_float prices1.(idx) in
      let bits2 = Int64.bits_of_float prices2.(idx) in
      price_hash := !price_hash lxor (Int64.to_int (Int64.logxor bits1 bits2))
    end
  done;
  
  let extended_key = Array.append key_params [| float_of_int !price_hash; float_of_int n |] in
  let cache_key = ParameterHash.hash extended_key in
  
  match ThreadSafeLRUCache.get backtest_cache cache_key with
  | Some result -> result
  | None ->
      let result = cpp_vectorized_backtest prices1 prices2 params in
      ThreadSafeLRUCache.put backtest_cache cache_key result;
      Printf.printf "✅ EXITING cpp_cached_vectorized_backtest() at %s\n" (get_current_time ());
      result

(* Cached objective evaluation *)
let cpp_cached_objective_evaluation params prices1 prices2 l1_ratio alpha kl_weight =
  let extended_key = Array.append params [| l1_ratio; alpha; kl_weight |] in
  let n = Array.length prices1 in
  let sample_size = min n 50 in
  let price_hash = ref 0 in
  
  for i = 0 to sample_size - 1 do
    if i * 5 < n then begin
      price_hash := !price_hash lxor (Hashtbl.hash prices1.(i * 5)) lxor (Hashtbl.hash prices2.(i * 5))
    end
  done;
  
  let final_key = Array.append extended_key [| float_of_int !price_hash |] in
  let cache_key = ParameterHash.hash final_key in
  
  match ThreadSafeLRUCache.get objective_cache cache_key with
  | Some result -> result
  | None ->
      let result = parallel_cross_validation prices1 prices2 params 3 l1_ratio alpha kl_weight in
      ThreadSafeLRUCache.put objective_cache cache_key result;
      result

(* Cache statistics *)
let cpp_print_cache_statistics () =
  Printf.printf "📊 ENTERING cpp_print_cache_statistics() at %s\n" (get_current_time ());
  let backtest_stats = ThreadSafeLRUCache.get_stats backtest_cache in
  let objective_stats = ThreadSafeLRUCache.get_stats objective_cache in
  let spread_stats = ThreadSafeLRUCache.get_stats spread_cache in
  
  Printf.printf "OCaml-Optimized Cache Statistics:\n";
  Printf.printf "Backtest Cache - Size: %d, Hit Rate: %.2f%%, Hits: %d, Misses: %d, Evictions: %d\n"
    backtest_stats.size (backtest_stats.hit_rate *. 100.0) backtest_stats.hits backtest_stats.misses backtest_stats.evictions;
  Printf.printf "Objective Cache - Size: %d, Hit Rate: %.2f%%, Hits: %d, Misses: %d, Evictions: %d\n"
    objective_stats.size (objective_stats.hit_rate *. 100.0) objective_stats.hits objective_stats.misses objective_stats.evictions;
  Printf.printf "Spread Cache - Size: %d, Hit Rate: %.2f%%, Hits: %d, Misses: %d, Evictions: %d\n"
    spread_stats.size (spread_stats.hit_rate *. 100.0) spread_stats.hits spread_stats.misses spread_stats.evictions;
  Printf.printf "✅ EXITING cpp_print_cache_statistics() at %s\n" (get_current_time ())

let cpp_clear_all_caches () =
  Printf.printf "🧹 ENTERING cpp_clear_all_caches() at %s\n" (get_current_time ());
  ThreadSafeLRUCache.clear backtest_cache;
  ThreadSafeLRUCache.clear objective_cache;
  ThreadSafeLRUCache.clear spread_cache;
  Printf.printf "✅ EXITING cpp_clear_all_caches() at %s\n" (get_current_time ())

(* Cache warm-up *)
let cpp_warm_up_caches prices1 prices2 =
  Printf.printf "🔥 ENTERING cpp_warm_up_caches() at %s\n" (get_current_time ());
  
  let common_params = [|
    [| 20.0; 2.0; 0.5; 10000.0; 0.001; 2.0; 1.0 |];
    [| 25.0; 2.5; 0.3; 15000.0; 0.001; 2.5; 1.5 |];
    [| 30.0; 1.5; 0.8; 20000.0; 0.0005; 1.8; 0.8 |];
    [| 15.0; 3.0; 0.2; 12000.0; 0.002; 3.0; 2.0 |];
    [| 35.0; 1.8; 0.6; 25000.0; 0.0015; 2.2; 1.2 |];
  |] in
  
  Printf.printf "Warming up OCaml-optimized caches with common parameter combinations...\n";
  
  Array.iter (fun params_array ->
    let trading_params = {
      lookback = int_of_float params_array.(0);
      z_entry = params_array.(1);
      z_exit = params_array.(2);
      position_size = int_of_float params_array.(3);
      transaction_cost = params_array.(4);
      profit_target = params_array.(5);
      stop_loss = params_array.(6);
    } in
    
    let _ = cpp_cached_vectorized_backtest prices1 prices2 trading_params in
    let _ = cpp_cached_objective_evaluation params_array prices1 prices2 0.7 0.02 0.15 in
    ()
  ) common_params;
  
  Printf.printf "OCaml cache warm-up complete.\n";
  cpp_print_cache_statistics ();
  Printf.printf "✅ EXITING cpp_warm_up_caches() at %s\n" (get_current_time ())

(* Optimization cache implementation *)
module OptimizationCache = struct
  type 'a t = {
    cache : (string, 'a) Hashtbl.t;
    max_size : int;
    mutable current_size : int;
    mutex : Mutex.t;
  }
  
  let create ?(max_size=10000) () =
    {
      cache = Hashtbl.create max_size;
      max_size = max_size;
      current_size = 0;
      mutex = Mutex.create ();
    }
  
  let get cache key =
    Mutex.lock cache.mutex;
    let result = try Some (Hashtbl.find cache.cache key) with Not_found -> None in
    Mutex.unlock cache.mutex;
    result
  
  let put cache key value =
    Mutex.lock cache.mutex;
    if not (Hashtbl.mem cache.cache key) then begin
      if cache.current_size >= cache.max_size then begin
        (* Simple eviction - remove random element *)
        let keys = Hashtbl.fold (fun k _ acc -> k :: acc) cache.cache [] in
        match keys with
        | oldest :: _ -> 
            Hashtbl.remove cache.cache oldest;
            cache.current_size <- cache.current_size - 1
        | [] -> ()
      end;
      cache.current_size <- cache.current_size + 1
    end;
    Hashtbl.replace cache.cache key value;
    Mutex.unlock cache.mutex
  
  let clear cache =
    Mutex.lock cache.mutex;
    Hashtbl.clear cache.cache;
    cache.current_size <- 0;
    Mutex.unlock cache.mutex
  
  let size cache =
    Mutex.lock cache.mutex;
    let s = cache.current_size in
    Mutex.unlock cache.mutex;
    s
end

(* Public interface functions *)
let cached_vectorized_backtest prices1 prices2 params =
  Printf.printf "💾 ENTERING cached_vectorized_backtest() at %s\n" (get_current_time ());
  let result = cpp_cached_vectorized_backtest prices1 prices2 params in
  let c_result = to_c_result result in
  Printf.printf "✅ EXITING cached_vectorized_backtest() at %s\n" (get_current_time ());
  c_result

let cached_objective_evaluation params prices1 prices2 l1_ratio alpha kl_weight =
  Printf.printf "🎯 ENTERING cached_objective_evaluation() at %s\n" (get_current_time ());
  let result = cpp_cached_objective_evaluation params prices1 prices2 l1_ratio alpha kl_weight in
  Printf.printf "✅ EXITING cached_objective_evaluation() at %s\n" (get_current_time ());
  result

let print_cache_statistics () =
  Printf.printf "📊 ENTERING print_cache_statistics() at %s\n" (get_current_time ());
  cpp_print_cache_statistics ();
  Printf.printf "✅ EXITING print_cache_statistics() at %s\n" (get_current_time ())

let clear_all_caches () =
  Printf.printf "🧹 ENTERING clear_all_caches() at %s\n" (get_current_time ());
  cpp_clear_all_caches ();
  Printf.printf "✅ EXITING clear_all_caches() at %s\n" (get_current_time ())

let warm_up_caches prices1 prices2 =
  Printf.printf "🔥 ENTERING warm_up_caches() at %s\n" (get_current_time ());
  cpp_warm_up_caches prices1 prices2;
  Printf.printf "✅ EXITING warm_up_caches() at %s\n" (get_current_time ())
