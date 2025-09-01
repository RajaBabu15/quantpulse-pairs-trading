(* QuantPulse Trading Engine - OCaml Interface *)
(* Equivalent to trading_engine.h *)

(* Core data structures *)
type price_data = {
  symbol1 : float array;
  symbol2 : float array;
  size : int;
}

type trading_parameters = {
  lookback : int;
  z_entry : float;
  z_exit : float;
  position_size : int;
  transaction_cost : float;
  profit_target : float;
  stop_loss : float;
}

type backtest_result = {
  final_pnl : float;
  sharpe_ratio : float;
  win_rate : float;
  max_drawdown : float;
  trade_returns : float array;
  num_trades : int;
}

type cv_fold_result = {
  objective_score : float;
  train_result : backtest_result;
  val_result : backtest_result;
  elasticnet_penalty : float;
  kl_penalty : float;
  stability_penalty : float;
}

type c_backtest_result = {
  total_return : float;
  sharpe_ratio : float;
  max_drawdown : float;
  num_trades : int;
  win_rate : float;
  profit_factor : float;
  avg_trade_return : float;
  volatility : float;
}

(* Thread pool management *)
module ThreadPool : sig
  type t
  val create : ?threads:int -> unit -> t
  val submit : t -> ('a -> 'b) -> 'a -> 'b Lwt.t
  val shutdown : t -> unit
  val thread_count : t -> int
end

(* Optimization cache *)
module OptimizationCache : sig
  type 'a t
  val create : ?max_size:int -> unit -> 'a t
  val get : 'a t -> string -> 'a option
  val put : 'a t -> string -> 'a -> unit
  val clear : 'a t -> unit
  val size : 'a t -> int
end

(* SIMD vectorized operations *)
module Simd : sig
  val vectorized_add : float array -> float array -> float array
  val vectorized_subtract : float array -> float array -> float array
  val vectorized_multiply : float array -> float array -> float array
  val vectorized_sum : float array -> float
  val vectorized_mean : float array -> float
  val vectorized_std : float array -> float -> float
end

(* Core functions *)
val calculate_spread_and_zscore : 
  float array -> float array -> int -> float array * float array

val vectorized_backtest : 
  float array -> float array -> trading_parameters -> c_backtest_result

val cached_vectorized_backtest : 
  float array -> float array -> trading_parameters -> c_backtest_result

val parallel_cross_validation : 
  float array -> float array -> float array -> int -> float -> float -> float -> float

val batch_parameter_optimization : 
  float array -> float array -> float array array -> c_backtest_result array

val backtest_trade_returns : 
  float array -> float array -> trading_parameters -> float array

val print_cache_statistics : unit -> unit
val clear_all_caches : unit -> unit
val warm_up_caches : float array -> float array -> unit

(* Constants *)
val epsilon : float
val cache_line_size : int
val max_threads : int
