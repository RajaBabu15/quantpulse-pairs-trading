(* Cross Validation and Parallel Optimization - OCaml Implementation *)
(* Equivalent to cross_validation.cpp *)

open Trading_engine
open Vectorized_math

(* Thread pool implementation using lightweight threads *)
module ThreadPool = struct
  type task = unit -> unit
  
  type t = {
    mutable workers : Thread.t array;
    task_queue : task Queue.t;
    queue_mutex : Mutex.t;
    queue_condition : Condition.t;
    mutable stop : bool;
    thread_count : int;
  }
  
  let create ?(threads=0) () =
    let actual_threads = if threads = 0 then 
      max 1 (min 32 (Sys.runtime_parameters () |> List.length))
    else min threads max_threads in
    {
      workers = [||];
      task_queue = Queue.create ();
      queue_mutex = Mutex.create ();
      queue_condition = Condition.create ();
      stop = false;
      thread_count = actual_threads;
    }
  
  let worker_loop pool =
    let rec loop () =
      Mutex.lock pool.queue_mutex;
      while Queue.is_empty pool.task_queue && not pool.stop do
        Condition.wait pool.queue_condition pool.queue_mutex
      done;
      
      if not pool.stop && not (Queue.is_empty pool.task_queue) then begin
        let task = Queue.take pool.task_queue in
        Mutex.unlock pool.queue_mutex;
        (try task () with _ -> ());
        loop ()
      end else begin
        Mutex.unlock pool.queue_mutex
      end
    in
    loop ()
  
  let start_workers pool =
    pool.workers <- Array.init pool.thread_count (fun _ ->
      Thread.create worker_loop pool
    )
  
  let submit pool f arg =
    let result = ref None in
    let task () = result := Some (f arg) in
    Mutex.lock pool.queue_mutex;
    Queue.push task pool.task_queue;
    Condition.signal pool.queue_condition;
    Mutex.unlock pool.queue_mutex;
    (* Simple blocking wait - in real implementation would use futures *)
    while !result = None do Thread.yield () done;
    match !result with Some v -> Lwt.return v | None -> failwith "Task failed"
  
  let shutdown pool =
    pool.stop <- true;
    Condition.broadcast pool.queue_condition;
    Array.iter Thread.join pool.workers
  
  let thread_count pool = pool.thread_count
end

(* Spread and z-score calculation *)
let calculate_spread_vectorized prices1 prices2 lookback =
  let n = Array.length prices1 in
  if n <> Array.length prices2 then
    failwith "Price arrays must have same length";
  
  let spread = Simd.vectorized_subtract prices1 prices2 in
  let z_scores = Array.make n 0.0 in
  
  for i = lookback to n - 1 do
    let window_start = i - lookback in
    let window = Array.sub spread window_start lookback in
    let mean = Simd.vectorized_mean window in
    let std_dev = Simd.vectorized_std window mean in
    z_scores.(i) <- if std_dev > epsilon then 
      (spread.(i) -. mean) /. std_dev 
    else 0.0
  done;
  
  (spread, z_scores)

(* Core backtest implementation *)
let cpp_vectorized_backtest prices1 prices2 params =
  Printf.printf "🔄 ENTERING cpp_vectorized_backtest() at %s\n" (get_current_time ());
  flush_all ();
  
  let n = Array.length prices1 in
  let spread, z_scores = calculate_spread_vectorized prices1 prices2 params.lookback in
  
  let pnl = ref 0.0 in
  let position = ref 0 in
  let entry_price = ref 0.0 in
  let trade_returns = ref [] in
  let max_dd = ref 0.0 in
  let peak = ref 0.0 in
  let winning_trades = ref 0 in
  let num_trades = ref 0 in
  
  for i = params.lookback to n - 1 do
    let current_z = z_scores.(i) in
    let current_spread = spread.(i) in
    
    if not (is_nan current_z) then begin
      if !position = 0 then begin
        (* Entry logic *)
        if current_z > params.z_entry then begin
          position := -1;
          entry_price := current_spread;
          pnl := !pnl -. (float_of_int params.position_size) *. params.transaction_cost
        end else if current_z < (-.params.z_entry) then begin
          position := 1;
          entry_price := current_spread;
          pnl := !pnl -. (float_of_int params.position_size) *. params.transaction_cost
        end
      end else begin
        (* Exit logic *)
        let spread_change = current_spread -. !entry_price in
        let unrealized_pnl = (float_of_int !position) *. spread_change *. (float_of_int params.position_size) in
        
        let exit_signal = 
          ((!position = 1 && spread_change > params.profit_target) ||
           (!position = -1 && spread_change < (-.params.profit_target))) ||
          ((!position = 1 && spread_change < (-.params.stop_loss)) ||
           (!position = -1 && spread_change > params.stop_loss)) ||
          (abs_float current_z < params.z_exit) in
        
        if exit_signal then begin
          let trade_pnl = unrealized_pnl in
          pnl := !pnl +. trade_pnl -. (float_of_int params.position_size) *. params.transaction_cost;
          trade_returns := trade_pnl :: !trade_returns;
          if trade_pnl > 0.0 then incr winning_trades;
          position := 0;
          entry_price := 0.0;
          incr num_trades
        end
      end;
      
      peak := max !peak !pnl;
      max_dd := max !max_dd (!peak -. !pnl)
    end
  done;
  
  let final_trade_returns = Array.of_list (List.rev !trade_returns) in
  let win_rate = if !num_trades > 0 then 
    (float_of_int !winning_trades) /. (float_of_int !num_trades) 
  else 0.0 in
  
  let sharpe_ratio = 
    if Array.length final_trade_returns > 0 then
      let mean_return = Simd.vectorized_mean final_trade_returns in
      let std_return = Simd.vectorized_std final_trade_returns mean_return in
      if std_return > epsilon then mean_return /. std_return else 0.0
    else 0.0 in
  
  Printf.printf "✅ EXITING cpp_vectorized_backtest() at %s\n" (get_current_time ());
  flush_all ();
  
  {
    final_pnl = !pnl;
    sharpe_ratio = sharpe_ratio;
    win_rate = win_rate;
    max_drawdown = !max_dd;
    trade_returns = final_trade_returns;
    num_trades = !num_trades;
  }

(* Cross validation fold evaluation *)
let evaluate_cv_fold_parallel prices params train_start train_end val_start val_end l1_ratio alpha kl_weight =
  let train_prices1 = Array.sub prices.symbol1 train_start (train_end - train_start) in
  let train_prices2 = Array.sub prices.symbol2 train_start (train_end - train_start) in
  let val_prices1 = Array.sub prices.symbol1 val_start (val_end - val_start) in
  let val_prices2 = Array.sub prices.symbol2 val_start (val_end - val_start) in
  
  let train_result = cpp_vectorized_backtest train_prices1 train_prices2 params in
  let val_result = cpp_vectorized_backtest val_prices1 val_prices2 params in
  
  (* Normalize parameters for penalty calculation *)
  let normalized_params = [|
    (float_of_int params.lookback -. 5.0) /. 55.0;
    (params.z_entry -. 0.5) /. 3.5;
    (params.z_exit -. 0.1) /. 1.9;
    (float_of_int params.position_size -. 1000.0) /. 49000.0;
    (params.transaction_cost -. 0.0001) /. 0.0049;
    (params.profit_target -. 1.5) /. 3.5;
    (params.stop_loss -. 0.5) /. 1.5;
  |] in
  
  let l1_penalty = Array.fold_left (fun acc x -> acc +. abs_float x) 0.0 normalized_params in
  let l2_penalty = Array.fold_left (fun acc x -> acc +. x *. x) 0.0 normalized_params in
  
  let elasticnet_penalty = alpha *. (l1_ratio *. l1_penalty +. (1.0 -. l1_ratio) *. l2_penalty) in
  let kl_penalty = kl_weight *. 0.1 in
  let stability_penalty = abs_float (train_result.sharpe_ratio -. val_result.sharpe_ratio) *. 0.1 in
  
  let primary_score = val_result.final_pnl /. 100000.0 +. val_result.sharpe_ratio in
  let secondary_score = val_result.win_rate *. 0.2 in
  let risk_adj = primary_score /. (abs_float val_result.max_drawdown /. 100000.0 +. 1.0) in
  let combined_score = risk_adj +. secondary_score in
  let objective_score = combined_score -. elasticnet_penalty -. kl_penalty -. stability_penalty in
  
  {
    objective_score = objective_score;
    train_result = train_result;
    val_result = val_result;
    elasticnet_penalty = elasticnet_penalty;
    kl_penalty = kl_penalty;
    stability_penalty = stability_penalty;
  }

(* Parallel cross validation implementation *)
let parallel_cross_validation_impl prices1 prices2 param_array n_splits l1_ratio alpha kl_weight =
  Printf.printf "🔧 ENTERING parallel_cross_validation_impl() at %s\n" (get_current_time ());
  
  let n = Array.length prices1 in
  let prices = { symbol1 = prices1; symbol2 = prices2; size = n } in
  let params = {
    lookback = int_of_float param_array.(0);
    z_entry = param_array.(1);
    z_exit = param_array.(2);
    position_size = int_of_float param_array.(3);
    transaction_cost = param_array.(4);
    profit_target = param_array.(5);
    stop_loss = param_array.(6);
  } in
  
  let fold_size = n / (n_splits + 1) in
  let splits = ref [] in
  
  for i = 0 to n_splits - 1 do
    let val_start = i * fold_size in
    let val_end = (i + 1) * fold_size in
    let train_start = val_end in
    let train_end = n in
    if train_end - train_start >= params.lookback + 10 then
      splits := ((train_start, train_end), (val_start, val_end)) :: !splits
  done;
  
  let results = List.map (fun ((train_start, train_end), (val_start, val_end)) ->
    evaluate_cv_fold_parallel prices params train_start train_end val_start val_end l1_ratio alpha kl_weight
  ) (List.rev !splits) in
  
  let total_score = List.fold_left (fun acc result -> acc +. result.objective_score) 0.0 results in
  let avg_score = if List.length results > 0 then total_score /. (float_of_int (List.length results)) else 0.0 in
  
  Printf.printf "✅ EXITING parallel_cross_validation_impl() at %s\n" (get_current_time ());
  avg_score

(* Convert to C-compatible result format *)
let to_c_result result =
  let avg_trade_return, volatility, profit_factor =
    if Array.length result.trade_returns > 0 then
      let avg = Simd.vectorized_mean result.trade_returns in
      let vol = Simd.vectorized_std result.trade_returns avg in
      let gross_pos = Array.fold_left (fun acc x -> if x >= 0.0 then acc +. x else acc) 0.0 result.trade_returns in
      let gross_neg = Array.fold_left (fun acc x -> if x < 0.0 then acc +. (-.x) else acc) 0.0 result.trade_returns in
      let pf = if gross_neg > epsilon then gross_pos /. gross_neg 
               else if gross_pos > 0.0 then infinity else 0.0 in
      (avg, vol, pf)
    else (0.0, 0.0, 0.0) in
  
  {
    total_return = result.final_pnl;
    sharpe_ratio = result.sharpe_ratio;
    max_drawdown = result.max_drawdown;
    num_trades = result.num_trades;
    win_rate = result.win_rate;
    profit_factor = profit_factor;
    avg_trade_return = avg_trade_return;
    volatility = volatility;
  }

(* Public interface functions *)
let calculate_spread_and_zscore prices1 prices2 lookback =
  Printf.printf "📊 ENTERING calculate_spread_and_zscore() at %s\n" (get_current_time ());
  let result = calculate_spread_vectorized prices1 prices2 lookback in
  Printf.printf "✅ EXITING calculate_spread_and_zscore() at %s\n" (get_current_time ());
  result

let vectorized_backtest prices1 prices2 params =
  Printf.printf "🚀 ENTERING vectorized_backtest() at %s\n" (get_current_time ());
  let result = cpp_vectorized_backtest prices1 prices2 params in
  let c_result = to_c_result result in
  Printf.printf "✅ EXITING vectorized_backtest() at %s\n" (get_current_time ());
  c_result

let backtest_trade_returns prices1 prices2 params =
  Printf.printf "📈 ENTERING backtest_trade_returns() at %s\n" (get_current_time ());
  let result = cpp_vectorized_backtest prices1 prices2 params in
  Printf.printf "✅ EXITING backtest_trade_returns() at %s\n" (get_current_time ());
  result.trade_returns

let parallel_cross_validation prices1 prices2 param_array n_splits l1_ratio alpha kl_weight =
  Printf.printf "🔧 ENTERING parallel_cross_validation() at %s\n" (get_current_time ());
  let result = parallel_cross_validation_impl prices1 prices2 param_array n_splits l1_ratio alpha kl_weight in
  Printf.printf "✅ EXITING parallel_cross_validation() at %s\n" (get_current_time ());
  result

let batch_parameter_optimization prices1 prices2 param_sets =
  Printf.printf "⚙️ ENTERING batch_parameter_optimization() at %s\n" (get_current_time ());
  let results = Array.map (fun params ->
    let trading_params = {
      lookback = int_of_float params.(0);
      z_entry = params.(1);
      z_exit = params.(2);
      position_size = int_of_float params.(3);
      transaction_cost = params.(4);
      profit_target = params.(5);
      stop_loss = params.(6);
    } in
    let result = cpp_vectorized_backtest prices1 prices2 trading_params in
    to_c_result result
  ) param_sets in
  Printf.printf "✅ EXITING batch_parameter_optimization() at %s\n" (get_current_time ());
  results
