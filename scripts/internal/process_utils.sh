#!/usr/bin/env bash
# Process cleanup shared by the local orchestrator and simulator retry wrapper.

_process_tree_pids() {
  local pid=$1 child
  while read -r child; do
    [[ -n "${child}" ]] || continue
    _process_tree_pids "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  printf '%s\n' "${pid}"
}

_process_identity() {
  # Start time survives exec (for example a conda wrapper becoming Python).
  LC_ALL=C ps -o lstart= -p "$1" 2>/dev/null || true
}

_stop_process_tree() {
  local pid=$1
  local attempt child index alive
  local -a pids=() identities=()
  # Snapshot before signalling: a parent may exit while a child ignores TERM
  # and becomes an orphan, so a later pgrep on the parent would miss it.
  while read -r child; do
    pids+=("${child}")
    identities+=("$(_process_identity "${child}")")
  done < <(_process_tree_pids "${pid}")
  for index in "${!pids[@]}"; do
    [[ -n "${identities[index]}" ]] || continue
    if [[ "$(_process_identity "${pids[index]}")" == "${identities[index]}" ]]; then
      kill -TERM "${pids[index]}" 2>/dev/null || true
    fi
  done
  for attempt in 1 2 3 4 5; do
    alive="false"
    for child in "${pids[@]}"; do
      if kill -0 "${child}" 2>/dev/null; then
        alive="true"
      fi
    done
    [[ "${alive}" == "true" ]] || break
    sleep 0.2
  done
  for index in "${!pids[@]}"; do
    [[ -n "${identities[index]}" ]] || continue
    # Do not signal an unrelated process if an exited PID was already reused.
    if [[ "$(_process_identity "${pids[index]}")" == "${identities[index]}" ]]; then
      kill -KILL "${pids[index]}" 2>/dev/null || true
    fi
  done
  wait "${pid}" 2>/dev/null || true
}
