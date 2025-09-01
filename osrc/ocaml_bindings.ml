(* OCaml Bindings - OCaml Implementation *)
(* Equivalent to python_bindings.cpp but for OCaml native interface *)

open Trading_engine
open Vectorized_math
open Cross_validation
open Performance_cache

(* JSON-like data structure for parameters *)
type param_dict = (string * float) list

(* Utility functions *)
let make_array n = Array.make n 0.0

let make_params param_dict =
  let get_param name default =
    try List.assoc name param_dict
    with Not_found -> default
  in
  {
    lookback = int_of_float (get_param "lookback" 20.0);
    z_entry = get_param "z_entry" 2.0;
    z_exit = get_param "z_exit" 0.5;
    position_size = int_of_float (get_param "position_size" 10000.0);
    transaction_cost = get_param "transaction_cost" 0.001;
    profit_target = get_param "profit_target" 2.0;
    stop_loss = get_param "stop_loss" 1.0;
  }

let to_param_dict result =
  [
    ("total_return", result.total_return);
    ("sharpe_ratio", result.sharpe_ratio);
    ("max_drawdown", result.max_drawdown);
    ("num_trades", float_of_int result.num_trades);
    ("win_rate", result.win_rate);
    ("profit_factor", result.profit_factor);
    ("avg_trade_return", result.avg_trade_return);
    ("volatility", result.volatility);
  ]

(* Main interface functions *)
module QuantPulseCore = struct
  
  (* SIMD vector operations *)
  let simd_vector_add a b =
    if Array.length a <> Array.length b then
      failwith "Arrays must have matching lengths"
    else
      Simd.vectorized_add a b
  
  let simd_vector_mean arr = Simd.vectorized_mean arr
  let simd_vector_std arr mean = Simd.vectorized_std arr mean
  
  (* Spread and z-score calculation *)
  let calculate_spread_and_zscore p1 p2 lookback =
    if Array.length p1 <> Array.length p2 then
      failwith "Price arrays must match";
    let spread, z_scores = calculate_spread_vectorized p1 p2 lookback in
    (spread, z_scores)
  
  (* Vectorized backtest *)
  let vectorized_backtest p1 p2 params ?(use_cache=true) () =
    if Array.length p1 <> Array.length p2 then
      failwith "Price arrays must match";
    let trading_params = make_params params in
    if use_cache then
      cached_vectorized_backtest p1 p2 trading_params
    else
      vectorized_backtest p1 p2 trading_params
  
  (* Parallel cross validation *)
  let parallel_cross_validation p1 p2 params ?(n_folds=3) ?(l1_ratio=0.7) ?(alpha=0.02) ?(kl_weight=0.15) () =
    if Array.length p1 <> Array.length p2 then
      failwith "Price arrays must match";
    parallel_cross_validation p1 p2 params n_folds l1_ratio alpha kl_weight
  
  (* Batch parameter optimization *)
  let batch_parameter_optimization p1 p2 param_sets =
    if Array.length p1 <> Array.length p2 then
      failwith "Price arrays must match";
    if Array.length param_sets = 0 then [||]
    else
      batch_parameter_optimization p1 p2 param_sets
  
  (* Trade returns *)
  let backtest_trade_returns p1 p2 params =
    if Array.length p1 <> Array.length p2 then
      failwith "Price arrays must match";
    let trading_params = make_params params in
    backtest_trade_returns p1 p2 trading_params
  
  (* Cache management *)
  let warm_up_caches p1 p2 =
    if Array.length p1 <> Array.length p2 then
      failwith "Price arrays must match";
    warm_up_caches p1 p2
  
  let print_cache_statistics () = print_cache_statistics ()
  let clear_all_caches () = clear_all_caches ()
end

(* Advanced functionality *)
module AdvancedAnalytics = struct
  
  (* Statistical analysis *)
  let calculate_sharpe_ratio returns =
    if Array.length returns = 0 then 0.0
    else
      let mean_return = Simd.vectorized_mean returns in
      let std_return = Simd.vectorized_std returns mean_return in
      if std_return > epsilon then mean_return /. std_return else 0.0
  
  let calculate_max_drawdown equity_curve =
    if Array.length equity_curve = 0 then 0.0
    else
      let peak = ref equity_curve.(0) in
      let max_dd = ref 0.0 in
      for i = 1 to Array.length equity_curve - 1 do
        peak := max !peak equity_curve.(i);
        let drawdown = !peak -. equity_curve.(i) in
        max_dd := max !max_dd drawdown
      done;
      !max_dd
  
  let calculate_sortino_ratio returns target_return =
    if Array.length returns = 0 then 0.0
    else
      let excess_returns = Array.map (fun r -> r -. target_return) returns in
      let mean_excess = Simd.vectorized_mean excess_returns in
      let downside_returns = Array.map (fun r -> if r < 0.0 then r *. r else 0.0) excess_returns in
      let downside_deviation = sqrt (Simd.vectorized_mean downside_returns) in
      if downside_deviation > epsilon then mean_excess /. downside_deviation else 0.0
  
  (* Risk metrics *)
  let calculate_value_at_risk returns confidence_level =
    if Array.length returns = 0 then 0.0
    else
      let sorted_returns = Array.copy returns in
      Array.sort compare sorted_returns;
      let index = int_of_float ((1.0 -. confidence_level) *. float_of_int (Array.length returns)) in
      if index < Array.length sorted_returns then sorted_returns.(index)
      else sorted_returns.(Array.length sorted_returns - 1)
  
  let calculate_expected_shortfall returns confidence_level =
    let var = calculate_value_at_risk returns confidence_level in
    let tail_returns = Array.fold_left (fun acc r ->
      if r <= var then r :: acc else acc
    ) [] returns in
    if List.length tail_returns = 0 then 0.0
    else
      let tail_array = Array.of_list tail_returns in
      Simd.vectorized_mean tail_array
  
  (* Portfolio metrics *)
  let calculate_calmar_ratio annual_return max_drawdown =
    if abs_float max_drawdown < epsilon then 0.0
    else annual_return /. abs_float max_drawdown
  
  let calculate_information_ratio excess_returns tracking_error =
    if abs_float tracking_error < epsilon then 0.0
    else
      let mean_excess = Simd.vectorized_mean excess_returns in
      mean_excess /. tracking_error
  
  (* Performance attribution *)
  let calculate_rolling_metrics returns window_size =
    let n = Array.length returns in
    if window_size >= n then [||]
    else
      let num_windows = n - window_size + 1 in
      let rolling_sharpe = Array.make num_windows 0.0 in
      let rolling_volatility = Array.make num_windows 0.0 in
      
      for i = 0 to num_windows - 1 do
        let window = Array.sub returns i window_size in
        let mean_ret = Simd.vectorized_mean window in
        let std_ret = Simd.vectorized_std window mean_ret in
        rolling_sharpe.(i) <- if std_ret > epsilon then mean_ret /. std_ret else 0.0;
        rolling_volatility.(i) <- std_ret
      done;
      
      (rolling_sharpe, rolling_volatility)
end

(* Optimization utilities *)
module OptimizationUtils = struct
  
  (* Parameter space exploration *)
  let generate_parameter_grid lookback_range z_entry_range z_exit_range =
    let results = ref [] in
    List.iter (fun lookback ->
      List.iter (fun z_entry ->
        List.iter (fun z_exit ->
          if z_exit < z_entry then
            results := [|
              float_of_int lookback; z_entry; z_exit; 10000.0; 0.001; 2.0; 1.0
            |] :: !results
        ) z_exit_range
      ) z_entry_range
    ) lookback_range;
    Array.of_list (List.rev !results)
  
  (* Genetic algorithm helpers *)
  let mutate_parameters params mutation_rate =
    Array.map (fun p ->
      if Random.float 1.0 < mutation_rate then
        p +. (Random.float 0.2 -. 0.1) *. p
      else p
    ) params
  
  let crossover_parameters parent1 parent2 =
    let child = Array.make (Array.length parent1) 0.0 in
    for i = 0 to Array.length parent1 - 1 do
      child.(i) <- if Random.bool () then parent1.(i) else parent2.(i)
    done;
    child
  
  (* Multi-objective optimization *)
  let pareto_dominance obj1 obj2 =
    let dominates = ref false in
    let dominated = ref false in
    for i = 0 to Array.length obj1 - 1 do
      if obj1.(i) > obj2.(i) then dominates := true;
      if obj1.(i) < obj2.(i) then dominated := true
    done;
    if !dominates && not !dominated then 1    (* obj1 dominates obj2 *)
    else if !dominated && not !dominates then -1  (* obj2 dominates obj1 *)
    else 0  (* non-dominated *)
  
  let find_pareto_front solutions objectives =
    let n = Array.length solutions in
    let pareto_set = ref [] in
    
    for i = 0 to n - 1 do
      let is_dominated = ref false in
      for j = 0 to n - 1 do
        if i <> j && pareto_dominance objectives.(j) objectives.(i) = 1 then
          is_dominated := true
      done;
      if not !is_dominated then
        pareto_set := (solutions.(i), objectives.(i)) :: !pareto_set
    done;
    
    Array.of_list !pareto_set
end

(* Export main interface *)
let () = 
  Printf.printf "QuantPulse OCaml Core Module Loaded\n";
  Printf.printf "Available modules: QuantPulseCore, AdvancedAnalytics, OptimizationUtils\n";
  flush_all ()

(* Example usage function *)
let example_usage () =
  (* Generate sample data *)
  let n = 1000 in
  let prices1 = Array.init n (fun i -> 100.0 +. 10.0 *. sin (float_of_int i /. 50.0) +. Random.float 2.0) in
  let prices2 = Array.init n (fun i -> 95.0 +. 12.0 *. sin (float_of_int i /. 55.0) +. Random.float 2.0) in
  
  (* Test basic functionality *)
  Printf.printf "🚀 Testing QuantPulse OCaml Core\n";
  
  (* Test SIMD operations *)
  let spread = QuantPulseCore.simd_vector_add prices1 (Array.map (fun x -> -.x) prices2) in
  let mean_spread = QuantPulseCore.simd_vector_mean spread in
  Printf.printf "Mean spread: %.4f\n" mean_spread;
  
  (* Test backtest *)
  let params = [("lookback", 20.0); ("z_entry", 2.0); ("z_exit", 0.5)] in
  let result = QuantPulseCore.vectorized_backtest prices1 prices2 params () in
  let result_dict = to_param_dict result in
  List.iter (fun (k, v) -> Printf.printf "%s: %.4f\n" k v) result_dict;
  
  (* Test advanced analytics *)
  let trade_returns = QuantPulseCore.backtest_trade_returns prices1 prices2 params in
  let sharpe = AdvancedAnalytics.calculate_sharpe_ratio trade_returns in
  Printf.printf "Calculated Sharpe: %.4f\n" sharpe;
  
  Printf.printf "✅ OCaml Core Tests Complete\n"
