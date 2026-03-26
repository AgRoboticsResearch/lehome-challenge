#!/bin/bash
# tmux development session for LeHome Challenge
# Supports dual environments: main (ACT/SmolVLA) and pi05

SESSION="lehome-dev"
ROOT="/home/admin/codes/lehome-challenge"

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION -c "$ROOT" -n main

# Window 0: Main environment (ACT/SmolVLA)
tmux send-keys -t $SESSION:0 "# Main environment: ACT, SmolVLA, existing workflows" C-m
tmux send-keys -t $SESSION:0 "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:0 "clear" C-m

# Window 1: π0.5 environment
tmux new-window -t $SESSION -n pi05 -c "$ROOT"
tmux send-keys -t $SESSION:pi05 "# π0.5 environment: Training & Development" C-m
if [ -d "$ROOT/.venv-pi" ]; then
    tmux send-keys -t $SESSION:pi05 "source .venv-pi/bin/activate" C-m
else
    tmux send-keys -t $SESSION:pi05 "# ⚠️ .venv-pi not found. Run: ./scripts/env/setup-pi.sh" C-m
fi
tmux send-keys -t $SESSION:pi05 "clear" C-m

# Window 2: Evaluation (main env by default)
tmux new-window -t $SESSION -n eval -c "$ROOT"
tmux send-keys -t $SESSION:eval "# Evaluation & Performance Comparison" C-m
tmux send-keys -t $SESSION:eval "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION:eval "clear" C-m

# Window 3: Shell / Git operations
tmux new-window -t $SESSION -n shell -c "$ROOT"
tmux send-keys -t $SESSION:shell "# Shell commands, Git operations" C-m
tmux send-keys -t $SESSION:shell "clear" C-m

# Select first window
tmux select-window -t $SESSION:0

# Attach to session
tmux attach -t $SESSION
