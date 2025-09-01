(* Vectorized Mathematical Operations - OCaml Implementation *)
(* Equivalent to vectorized_math.cpp *)

open Trading_engine

(* Timing utilities *)
let get_current_time () =
  let time = Unix.time () in
  let tm = Unix.localtime time in
  Printf.sprintf "%02d:%02d:%02d" tm.tm_hour tm.tm_min tm.tm_sec

(* SIMD-style vectorized operations using OCaml arrays *)
module Simd = struct
  
  let vectorized_add a b =
    if Array.length a <> Array.length b then
      failwith "Array lengths must match"
    else
      Array.mapi (fun i x -> x +. b.(i)) a
  
  let vectorized_subtract a b =
    if Array.length a <> Array.length b then
      failwith "Array lengths must match"
    else
      Array.mapi (fun i x -> x -. b.(i)) a
  
  let vectorized_multiply a b =
    if Array.length a <> Array.length b then
      failwith "Array lengths must match"
    else
      Array.mapi (fun i x -> x *. b.(i)) a
  
  let vectorized_sum arr =
    Array.fold_left (+.) 0.0 arr
  
  let vectorized_mean arr =
    let n = Array.length arr in
    if n = 0 then 0.0
    else vectorized_sum arr /. (float_of_int n)
  
  let vectorized_std arr mean =
    let n = Array.length arr in
    if n <= 1 then 0.0
    else begin
      let sum_sq = Array.fold_left (fun acc x ->
        let diff = x -. mean in
        acc +. diff *. diff
      ) 0.0 arr in
      sqrt (sum_sq /. (float_of_int (n - 1)))
    end
end

(* Optimized array operations with functional programming *)
let array_map2 f a b =
  if Array.length a <> Array.length b then
    failwith "Array lengths must match"
  else
    Array.mapi (fun i x -> f x b.(i)) a

let array_fold_left2 f init a b =
  if Array.length a <> Array.length b then
    failwith "Array lengths must match"
  else
    let acc = ref init in
    for i = 0 to Array.length a - 1 do
      acc := f !acc a.(i) b.(i)
    done;
    !acc

(* Parallel array operations using Domainslib when available *)
let parallel_map f arr =
  (* Fallback to sequential map for compatibility *)
  Array.map f arr

let parallel_fold_left f init arr =
  (* Fallback to sequential fold for compatibility *)
  Array.fold_left f init arr

(* Functional replacements for SIMD operations *)
let simd_vector_add a b result =
  Array.blit (Simd.vectorized_add a b) 0 result 0 (Array.length a)

let simd_vector_subtract a b result =
  Array.blit (Simd.vectorized_subtract a b) 0 result 0 (Array.length a)

let simd_vector_multiply a b result =
  Array.blit (Simd.vectorized_multiply a b) 0 result 0 (Array.length a)

let simd_vector_sum arr = Simd.vectorized_sum arr
let simd_vector_mean arr = Simd.vectorized_mean arr
let simd_vector_std arr mean = Simd.vectorized_std arr mean

(* Mathematical utilities *)
let is_nan x = x <> x
let is_finite x = not (is_nan x) && x <> infinity && x <> neg_infinity

let safe_divide x y =
  if abs_float y < epsilon then 0.0
  else x /. y

(* Statistical functions *)
let calculate_correlation x y =
  let n = Array.length x in
  if n <> Array.length y || n < 2 then 0.0
  else begin
    let mean_x = Simd.vectorized_mean x in
    let mean_y = Simd.vectorized_mean y in
    let sum_xy = ref 0.0 in
    let sum_xx = ref 0.0 in
    let sum_yy = ref 0.0 in
    
    for i = 0 to n - 1 do
      let dx = x.(i) -. mean_x in
      let dy = y.(i) -. mean_y in
      sum_xy := !sum_xy +. dx *. dy;
      sum_xx := !sum_xx +. dx *. dx;
      sum_yy := !sum_yy +. dy *. dy
    done;
    
    let denom = sqrt (!sum_xx *. !sum_yy) in
    if abs_float denom < epsilon then 0.0
    else !sum_xy /. denom
  end

let rolling_window f window_size arr =
  let n = Array.length arr in
  if window_size <= 0 || window_size > n then [||]
  else begin
    let result = Array.make (n - window_size + 1) 0.0 in
    for i = 0 to n - window_size do
      let window = Array.sub arr i window_size in
      result.(i) <- f window
    done;
    result
  end

(* Optimization helpers *)
let clip_value value min_val max_val =
  if value < min_val then min_val
  else if value > max_val then max_val
  else value

let normalize_array arr =
  let mean = Simd.vectorized_mean arr in
  let std = Simd.vectorized_std arr mean in
  if std < epsilon then Array.make (Array.length arr) 0.0
  else Array.map (fun x -> (x -. mean) /. std) arr

let rescale_array arr new_min new_max =
  let old_min = Array.fold_left min infinity arr in
  let old_max = Array.fold_left max neg_infinity arr in
  let old_range = old_max -. old_min in
  let new_range = new_max -. new_min in
  if abs_float old_range < epsilon then arr
  else Array.map (fun x -> 
    new_min +. (x -. old_min) *. new_range /. old_range
  ) arr

(* Performance logging *)
let time_function name f =
  let start_time = get_current_time () in
  Printf.printf "🔄 ENTERING %s() at %s\n" name start_time;
  flush_all ();
  let result = f () in
  let end_time = get_current_time () in
  Printf.printf "✅ EXITING %s() at %s\n" name end_time;
  flush_all ();
  result
