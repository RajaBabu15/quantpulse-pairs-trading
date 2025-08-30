# FPGA Low-Latency Design for Rolling Z-Score and Pairwise Correlation

This document outlines an FPGA offload for the latency-critical rolling statistics used by the HFT engine.

Target platform
- Typical: PCIe-attached FPGA (e.g., Xilinx Alveo U250/U280 or Intel Agilex)
- Host OS: Prefer Linux for kernel-bypass networking (DPDK/Onload/RDMA) and XDMA drivers. macOS is not suitable for production deployment with PCIe FPGAs.
- Link: PCIe Gen3/4 x8 or x16
- Network: 10G/25G/100G (optional inline feed handling)

Workload summary
- Inputs: Rolling ring-buffers per symbol (double), per-symbol write indices
- Per tick batch: Update ring buffers and compute:
  - Incremental z-score-per-pair (using sums and sumsq for the spread)
  - Incremental correlation-per-pair (using sx, sy, sxx, syy, sxy)
- Access pattern: O(1) per pair per tick due to rolling updates

High-level architecture
1) Ingress
   - Option A: Host->FPGA DMA (XDMA) of updated samples and new write indices
   - Option B: Inline network feed parser on FPGA -> ring buffer update without host involvement
2) Ring buffers (on-chip)
   - BRAM/URAM banks partitioned by symbol for parallel access
   - Write index register file per symbol
3) Rolling-state caches
   - For each pair i: sx[i], sy[i], sxx[i], syy[i], sxy[i] (float64) for correlation
   - For each pair i: s[i], ssq[i] for z-score spread
4) Update pipeline (per tick or mini-batch)
   - Lookup indices: new = w-1, old = w-lookback
   - Read xa_new, xb_new, xa_old, xb_old from ring buffers
   - Update state in a few cycles, compute z and corr
5) Egress
   - Return signals (int8) and correlation (float64) vectors via DMA, or expose in BAR-mapped registers for polling

Scheduling and parallelism
- Partition pairs across parallel PEs (processing elements)
- Use dual-port BRAM for simultaneous reads of symbol A and B
- Pipeline stage latency: O(10–30 cycles) -> sub-100ns internal; constrained by IO (PCIe/Network)

Host interface API
- Control: IOCTL or userspace mapping for lookback, pairs, thresholds
- Data: DMA queues for (symbol_id, price) updates and results

Notes for Apple silicon/macOS
- macOS is not a supported dev host for PCIe FPGA cards. Use a Linux host (e.g., Ubuntu LTS) for building and validating FPGA bitstreams and drivers.

Example HLS kernel (Xilinx Vitis HLS, C++)
- Simplified combinational step for one pair update (ignores DMA wiring and multiple PEs)

