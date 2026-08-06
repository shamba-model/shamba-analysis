#!/bin/bash
# Backfill morris_run.log files from an already-run batch into a log_files/
# subfolder of the results dir. Only needed for batches run before
# run_morris_batch.py started copying logs itself; new runs don't need this.
#
# Usage: shamba/model/morris/collect_batch_logs.sh [results_dir]

RESULTS_DIR="${1:-shamba/model/morris/results/morris_batch}"

mkdir -p "${RESULTS_DIR}/log_files"

for f in shamba/projects/morris_batch_*/morris_run.log; do
  cp "$f" "${RESULTS_DIR}/log_files/morris_run_$(basename "$(dirname "$f")").log"
done
