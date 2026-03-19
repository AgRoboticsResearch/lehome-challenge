#!/bin/bash

# =============================================================================
# LeHome Challenge - Multi-Checkpoint Evaluation Script
# =============================================================================
# Automatically evaluates multiple checkpoints and generates comparison report
#
# Usage:
#   ./scripts/eval_checkpoints.sh --model smolvla_pant_short \
#       --dataset_root Datasets/example/pant_short_merged \
#       --garment_type pant_short
# =============================================================================

set -e

# Default values
NUM_EPISODES=5
MAX_STEPS=600
OUTPUT_DIR="outputs/eval_reports"
TASK_DESCRIPTION="fold the garment on the table"
CHECKPOINT_MIN=""
CHECKPOINT_MAX=""
CHECKPOINT_LAST=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset_root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --garment_type)
            GARMENT_TYPE="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task_description)
            TASK_DESCRIPTION="$2"
            shift 2
            ;;
        --checkpoint_min)
            CHECKPOINT_MIN="$2"
            shift 2
            ;;
        --checkpoint_max)
            CHECKPOINT_MAX="$2"
            shift 2
            ;;
        --checkpoint_last)
            CHECKPOINT_LAST=true
            shift
            ;;
        --help|-h)
            cat << 'EOF'
LeHome Challenge - Multi-Checkpoint Evaluation

Usage:
  ./scripts/eval_checkpoints.sh --model <model_name> --dataset_root <path> --garment_type <type>

Required Arguments:
  --model           Model directory name (e.g., smolvla_pant_short)
  --dataset_root    Dataset path for metadata
  --garment_type    Garment category (top_long, top_short, pant_long, pant_short)

Optional Arguments:
  --num_episodes      Episodes per checkpoint (default: 5)
  --max_steps         Max steps per episode (default: 600)
  --output_dir        Report output directory (default: outputs/eval_reports)
  --task_description  Task description for VLA models (default: "fold the garment on the table")
  --checkpoint_min    Minimum checkpoint step to evaluate (e.g., 10000)
  --checkpoint_max    Maximum checkpoint step to evaluate (e.g., 20000)
  --checkpoint_last   Only evaluate the 'last' symlink
  --help              Show this help message

Examples:
  # Evaluate all checkpoints
  ./scripts/eval_checkpoints.sh \\
      --model smolvla_pant_short \\
      --dataset_root Datasets/example/pant_short_merged \\
      --garment_type pant_short

  # Evaluate checkpoints in range
  ./scripts/eval_checkpoints.sh \\
      --model smolvla_pant_short \\
      --dataset_root Datasets/example/pant_short_merged \\
      --garment_type pant_short \\
      --checkpoint_min 10000 \\
      --checkpoint_max 16000

  # Only evaluate latest checkpoint
  ./scripts/eval_checkpoints.sh \\
      --model smolvla_pant_short \\
      --dataset_root Datasets/example/pant_short_merged \\
      --garment_type pant_short \\
      --checkpoint_last
EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Error: --model is required${NC}"
    exit 1
fi
if [[ -z "$DATASET_ROOT" ]]; then
    echo -e "${RED}Error: --dataset_root is required${NC}"
    exit 1
fi
if [[ -z "$GARMENT_TYPE" ]]; then
    echo -e "${RED}Error: --garment_type is required${NC}"
    exit 1
fi

# Setup paths
CHECKPOINT_DIR="outputs/train/$MODEL/checkpoints"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_NAME="${MODEL}_${TIMESTAMP}"
REPORT_DIR="$OUTPUT_DIR/$REPORT_NAME"
LOG_DIR="$REPORT_DIR/logs"

# Create output directories
mkdir -p "$REPORT_DIR"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  LeHome Challenge - Multi-Checkpoint Evaluation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📁 Model:${NC} $MODEL"
echo -e "${YELLOW}📂 Checkpoint directory:${NC} $CHECKPOINT_DIR"

# Find checkpoints
if [[ "$CHECKPOINT_LAST" == true ]]; then
    # Only evaluate the 'last' symlink
    if [[ -L "$CHECKPOINT_DIR/last" ]]; then
        LAST_TARGET=$(readlink "$CHECKPOINT_DIR/last")
        CHECKPOINTS=("$CHECKPOINT_DIR/$LAST_TARGET")
        echo -e "${YELLOW}📌 Mode:${NC} Evaluating 'last' symlink only"
    else
        echo -e "${RED}Error: 'last' symlink not found in $CHECKPOINT_DIR${NC}"
        exit 1
    fi
else
    # Find all numeric checkpoint directories
    CHECKPOINTS=()
    while IFS= read -r -d '' path; do
        CHECKPOINTS+=("$path")
    done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "[0-9]*" -print0 | sort -z)

    # Apply min/max filters
    FILTERED_CHECKPOINTS=()
    for checkpoint in "${CHECKPOINTS[@]}"; do
        step=$(basename "$checkpoint")
        step_num=$((10#$step))  # Convert to number, removing leading zeros

        # Check min filter
        if [[ -n "$CHECKPOINT_MIN" ]]; then
            min_num=$((10#$CHECKPOINT_MIN))
            if (( step_num < min_num )); then
                continue
            fi
        fi

        # Check max filter
        if [[ -n "$CHECKPOINT_MAX" ]]; then
            max_num=$((10#$CHECKPOINT_MAX))
            if (( step_num > max_num )); then
                continue
            fi
        fi

        FILTERED_CHECKPOINTS+=("$checkpoint")
    done
    CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

# Check if any checkpoints found
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo -e "${RED}Error: No checkpoints found in $CHECKPOINT_DIR${NC}"
    exit 1
fi

# Display found checkpoints
echo -e "${YELLOW}📂 Checkpoints found (${#CHECKPOINTS[@]}):${NC}"
for checkpoint in "${CHECKPOINTS[@]}"; do
    step=$(basename "$checkpoint")
    echo "   - $step"
done
echo ""

# Display evaluation config
echo -e "${YELLOW}⚙️  Configuration:${NC}"
echo "   Dataset: $DATASET_ROOT"
echo "   Garment Type: $GARMENT_TYPE"
echo "   Episodes: $NUM_EPISODES"
echo "   Max Steps: $MAX_STEPS"
echo "   Task Description: $TASK_DESCRIPTION"
echo ""
echo -e "${YELLOW}📄 Report will be saved to:${NC} $REPORT_DIR"
echo ""

# Ask for confirmation
read -p "Start evaluation? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Store results for report
declare -A RESULTS_SUCCESS_RATE
declare -A RESULTS_SUCCESS_COUNT
declare -A RESULTS_AVG_RETURN
declare -A RESULTS_AVG_LENGTH
declare -A RESULTS_DURATION
declare -A RESULTS_PER_GARMENT  # New: stores per-garment data as JSON-like string
CHECKPOINT_STEPS=()

# Run evaluation for each checkpoint
total_start_time=$(date +%s)

for checkpoint in "${CHECKPOINTS[@]}"; do
    step=$(basename "$checkpoint")
    CHECKPOINT_STEPS+=("$step")

    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Evaluating checkpoint: $step${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"

    checkpoint_start_time=$(date +%s)

    # Run evaluation
    LOG_FILE="$LOG_DIR/${step}.log"
    python -m scripts.eval \
        --policy_type lerobot \
        --policy_path "$checkpoint/pretrained_model" \
        --dataset_root "$DATASET_ROOT" \
        --garment_type "$GARMENT_TYPE" \
        --num_episodes "$NUM_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --enable_cameras \
        --device cpu \
        --headless \
        --task_description "$TASK_DESCRIPTION" \
        2>&1 | tee "$LOG_FILE"

    checkpoint_end_time=$(date +%s)
    duration=$((checkpoint_end_time - checkpoint_start_time))
    duration_min=$((duration / 60))
    duration_sec=$((duration % 60))
    DURATION_STR="${duration_min}m ${duration_sec}s"
    RESULTS_DURATION[$step]="$DURATION_STR"

    # Parse results from log
    # Look for "Success Rate: XX.XX% (N/M)" pattern
    SUCCESS_RATE=$(grep -o "Success Rate: [0-9.]*%" "$LOG_FILE" | tail -1 | grep -o "[0-9.]*" || echo "0")
    SUCCESS_COUNT=$(grep -o "Success Rate: [0-9.]*% ([0-9]*/[0-9]*)" "$LOG_FILE" | tail -1 | grep -o "([0-9]*/[0-9]*)" | tr -d '()' || echo "0/$NUM_EPISODES")
    AVG_RETURN=$(grep -o "Avg Return: [0-9.]*" "$LOG_FILE" | tail -1 | grep -o "[0-9.]*" || echo "0")
    AVG_LENGTH=$(grep -o "Avg Length: [0-9.]*" "$LOG_FILE" | tail -1 | grep -o "[0-9.]*" || echo "0")

    RESULTS_SUCCESS_RATE[$step]="$SUCCESS_RATE"
    RESULTS_SUCCESS_COUNT[$step]="$SUCCESS_COUNT"
    RESULTS_AVG_RETURN[$step]="$AVG_RETURN"
    RESULTS_AVG_LENGTH[$step]="$AVG_LENGTH"

    # Parse per-garment breakdown from log
    # Format: "2026-03-13 13:01:39 - scripts.utils.evaluation - INFO -   Top_Long_Seen_0: Success Rate = 80.00%, Avg Return = 126.68"
    PER_GARMENT_DATA=""
    while IFS= read -r line; do
        # Extract garment name - look for pattern like "Top_Long_Seen_0:" or "Pant_Long_Unseen_1:"
        garment_name=$(echo "$line" | grep -oE '(Top|Pant)_[A-Za-z]+_[A-Za-z]+_[0-9]+' | head -1)
        garment_rate=$(echo "$line" | sed -n 's/.*Success Rate = \([0-9.]*\)%.*/\1/p')
        garment_return=$(echo "$line" | sed -n 's/.*Avg Return = \([0-9.]*\)/\1/p')
        if [[ -n "$garment_name" && -n "$garment_rate" ]]; then
            if [[ -n "$PER_GARMENT_DATA" ]]; then
                PER_GARMENT_DATA="${PER_GARMENT_DATA}|${garment_name}:${garment_rate},${garment_return}"
            else
                PER_GARMENT_DATA="${garment_name}:${garment_rate},${garment_return}"
            fi
        fi
    done < <(grep "Success Rate = " "$LOG_FILE")
    RESULTS_PER_GARMENT[$step]="$PER_GARMENT_DATA"

    echo ""
    echo -e "${YELLOW}📊 Checkpoint $step Results:${NC}"
    echo "   Success Rate: ${SUCCESS_RATE}%"
    echo "   Success Count: ${SUCCESS_COUNT}"
    echo "   Avg Return: ${AVG_RETURN}"
    echo "   Avg Length: ${AVG_LENGTH}"
    echo "   Duration: ${DURATION_STR}"
done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
total_duration_min=$((total_duration / 60))
total_duration_sec=$((total_duration % 60))

# Find best checkpoint
BEST_STEP=""
BEST_RATE=0
for step in "${CHECKPOINT_STEPS[@]}"; do
    rate=$(echo "${RESULTS_SUCCESS_RATE[$step]}" | tr -d '.' || echo "0")
    rate_int=$((10#$rate))
    if (( rate_int > BEST_RATE )); then
        BEST_RATE=$rate_int
        BEST_STEP=$step
    fi
done

# Generate Markdown report
REPORT_FILE="$REPORT_DIR/report.md"
cat > "$REPORT_FILE" << EOF
# Evaluation Report: $MODEL

**Generated:** $(date +"%Y-%m-%d %H:%M:%S")
**Dataset:** \`$DATASET_ROOT\`
**Garment Type:** \`$GARMENT_TYPE\`
**Episodes per checkpoint:** $NUM_EPISODES
**Total Duration:** ${total_duration_min}m ${total_duration_sec}s

## Summary Table

| Checkpoint | Success Rate | Success Count | Avg Return | Avg Length | Duration |
|------------|-------------|---------------|------------|------------|----------|
EOF

for step in "${CHECKPOINT_STEPS[@]}"; do
    rate="${RESULTS_SUCCESS_RATE[$step]}"
    count="${RESULTS_SUCCESS_COUNT[$step]}"
    ret="${RESULTS_AVG_RETURN[$step]}"
    len="${RESULTS_AVG_LENGTH[$step]}"
    dur="${RESULTS_DURATION[$step]}"

    # Bold the best checkpoint
    if [[ "$step" == "$BEST_STEP" ]]; then
        echo "| **$step** | **${rate}%** | **$count** | **$ret** | **$len** | $dur |" >> "$REPORT_FILE"
    else
        echo "| $step | ${rate}% | $count | $ret | $len | $dur |" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

## Best Checkpoint

**$BEST_STEP** with ${RESULTS_SUCCESS_RATE[$BEST_STEP]}% success rate

## Per-Garment Breakdown (Best Checkpoint: $BEST_STEP)

| Garment | Success Rate | Avg Return |
|---------|-------------|------------|
EOF

# Add per-garment rows for best checkpoint
if [[ -n "${RESULTS_PER_GARMENT[$BEST_STEP]}" ]]; then
    IFS='|' read -ra GARMENTS <<< "${RESULTS_PER_GARMENT[$BEST_STEP]}"
    for garment_data in "${GARMENTS[@]}"; do
        IFS=':' read -r garment_name garment_stats <<< "$garment_data"
        IFS=',' read -r garment_rate garment_return <<< "$garment_stats"
        echo "| $garment_name | ${garment_rate}% | $garment_return |" >> "$REPORT_FILE"
    done
else
    echo "| *No per-garment data available* | - | - |" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## All Checkpoints Per-Garment Comparison

EOF

# Generate comparison table for all checkpoints
# First, collect all garment names (using temp file for bash 3.x compatibility)
GARMENT_NAMES_FILE=$(mktemp)
for step in "${CHECKPOINT_STEPS[@]}"; do
    if [[ -n "${RESULTS_PER_GARMENT[$step]}" ]]; then
        IFS='|' read -ra GARMENTS <<< "${RESULTS_PER_GARMENT[$step]}"
        for garment_data in "${GARMENTS[@]}"; do
            IFS=':' read -r garment_name _ <<< "$garment_data"
            echo "$garment_name" >> "$GARMENT_NAMES_FILE"
        done
    fi
done
SORTED_GARMENTS=($(sort -u "$GARMENT_NAMES_FILE"))
rm -f "$GARMENT_NAMES_FILE"

# Build header row
HEADER="| Garment |"
for step in "${CHECKPOINT_STEPS[@]}"; do
    HEADER="$HEADER $step |"
done
echo "$HEADER" >> "$REPORT_FILE"

# Build separator row
SEPARATOR="|---------|"
for step in "${CHECKPOINT_STEPS[@]}"; do
    dashes=$(printf '%0.s-' $(seq 1 $((${#step} + 2))))
    SEPARATOR="$SEPARATOR$dashes|"
done
echo "$SEPARATOR" >> "$REPORT_FILE"

# Build data rows
for garment in "${SORTED_GARMENTS[@]}"; do
    ROW="| $garment |"
    for step in "${CHECKPOINT_STEPS[@]}"; do
        rate="-"
        if [[ -n "${RESULTS_PER_GARMENT[$step]}" ]]; then
            IFS='|' read -ra GARMENTS <<< "${RESULTS_PER_GARMENT[$step]}"
            for garment_data in "${GARMENTS[@]}"; do
                IFS=':' read -r gname gstats <<< "$garment_data"
                if [[ "$gname" == "$garment" ]]; then
                    IFS=',' read -r grate _ <<< "$gstats"
                    rate="${grate}%"
                    break
                fi
            done
        fi
        ROW="$ROW $rate |"
    done
    echo "$ROW" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

## Logs

Individual checkpoint logs available at: \`$REPORT_DIR/logs/\`

-EOF
EOF

# Generate visualization if matplotlib is available
PLOT_FILE="$REPORT_DIR/success_rate.png"
HEATMAP_FILE="$REPORT_DIR/per_garment_heatmap.png"
if python -c "import matplotlib" 2>/dev/null; then
    # Build Python list strings properly with quotes
    STEPS_PY=$(printf '%s,' "${CHECKPOINT_STEPS[@]}" | sed 's/,$//')
    RATES_PY=$(printf '%s,' "${RESULTS_SUCCESS_RATE[@]}" | sed 's/,$//')

    python << PYEOF
import matplotlib.pyplot as plt
import numpy as np
import os

steps = "$STEPS_PY".split(',')
rates_str = "$RATES_PY".split(',')
rates = [float(r) for r in rates_str]

# Bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(steps, rates, color='#4CAF50', edgecolor='black')

# Highlight best checkpoint
best_idx = rates.index(max(rates))
bars[best_idx].set_color('#FF5722')

plt.xlabel('Checkpoint Step', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.title('Success Rate by Checkpoint: $MODEL', fontsize=14)
plt.ylim(0, 100)
plt.xticks(rotation=45)

# Add value labels on bars
for bar, rate in zip(bars, rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('$PLOT_FILE', dpi=150)
print(f'Saved plot to $PLOT_FILE')
plt.close()
PYEOF

    # Generate per-garment heatmap using data file
    HEATMAP_DATA_FILE="$REPORT_DIR/heatmap_data.csv"
    echo "garment,${CHECKPOINT_STEPS[*]}" | tr ' ' ',' > "$HEATMAP_DATA_FILE"
    for garment in "${SORTED_GARMENTS[@]}"; do
        row="$garment"
        for step in "${CHECKPOINT_STEPS[@]}"; do
            rate="0"
            if [[ -n "${RESULTS_PER_GARMENT[$step]}" ]]; then
                IFS='|' read -ra GARMENTS <<< "${RESULTS_PER_GARMENT[$step]}"
                for garment_data in "${GARMENTS[@]}"; do
                    IFS=':' read -r gname gstats <<< "$garment_data"
                    if [[ "$gname" == "$garment" ]]; then
                        IFS=',' read -r grate _ <<< "$gstats"
                        rate="$grate"
                        break
                    fi
                done
            fi
            row="$row,$rate"
        done
        echo "$row" >> "$HEATMAP_DATA_FILE"
    done

    python << PYEOF
import matplotlib.pyplot as plt
import numpy as np
import csv

# Read heatmap data
with open('$HEATMAP_DATA_FILE', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    checkpoint_steps = header[1:]
    garments = []
    matrix = []
    for row in reader:
        garments.append(row[0])
        matrix.append([float(x) for x in row[1:]])

    matrix = np.array(matrix)

    if matrix.size > 0:
        fig, ax = plt.subplots(figsize=(max(10, len(checkpoint_steps) * 1.5), max(6, len(garments) * 0.5)))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(np.arange(len(checkpoint_steps)))
        ax.set_yticks(np.arange(len(garments)))
        ax.set_xticklabels(checkpoint_steps, rotation=45, ha='right')
        ax.set_yticklabels(garments)

        # Add text annotations
        for i in range(len(garments)):
            for j in range(len(checkpoint_steps)):
                val = matrix[i, j]
                text = ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                              color='white' if val < 50 else 'black', fontsize=9)

        ax.set_title('Per-Garment Success Rate Heatmap: $MODEL', fontsize=14)
        ax.set_xlabel('Checkpoint', fontsize=12)
        ax.set_ylabel('Garment', fontsize=12)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate (%)', fontsize=11)

        plt.tight_layout()
        plt.savefig('$HEATMAP_FILE', dpi=150)
        print(f'Saved heatmap to $HEATMAP_FILE')
        plt.close()
PYEOF

    # Update report to include images
    if [[ -f "$PLOT_FILE" ]]; then
        sed -i '' 's/-EOF//' "$REPORT_FILE"
        cat >> "$REPORT_FILE" << EOF

## Visualization

### Overall Success Rate

![Success Rate by Checkpoint](success_rate.png)
EOF
        if [[ -f "$HEATMAP_FILE" ]]; then
            cat >> "$REPORT_FILE" << EOF

### Per-Garment Heatmap

![Per-Garment Success Rate Heatmap](per_garment_heatmap.png)
EOF
        fi
        echo "" >> "$REPORT_FILE"
    fi
fi

# Final summary
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Evaluation Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}📊 Final Results:${NC}"
echo ""
printf "  %-12s %-14s %-14s %-12s %-12s\n" "Checkpoint" "Success Rate" "Success Count" "Avg Return" "Avg Length"
echo "  ──────────────────────────────────────────────────────────────────"
for step in "${CHECKPOINT_STEPS[@]}"; do
    if [[ "$step" == "$BEST_STEP" ]]; then
        printf "  ${GREEN}%-12s %-14s %-14s %-12s %-12s${NC}\n" "$step" "${RESULTS_SUCCESS_RATE[$step]}%" "${RESULTS_SUCCESS_COUNT[$step]}" "${RESULTS_AVG_RETURN[$step]}" "${RESULTS_AVG_LENGTH[$step]}"
    else
        printf "  %-12s %-14s %-14s %-12s %-12s\n" "$step" "${RESULTS_SUCCESS_RATE[$step]}%" "${RESULTS_SUCCESS_COUNT[$step]}" "${RESULTS_AVG_RETURN[$step]}" "${RESULTS_AVG_LENGTH[$step]}"
    fi
done
echo ""
echo -e "${GREEN}🏆 Best Checkpoint: $BEST_STEP (${RESULTS_SUCCESS_RATE[$BEST_STEP]}% success rate)${NC}"
echo ""
echo -e "${YELLOW}📄 Report saved to:${NC} $REPORT_FILE"
echo -e "${YELLOW}📁 Logs saved to:${NC} $LOG_DIR"
if [[ -f "$PLOT_FILE" ]]; then
    echo -e "${YELLOW}📈 Plot saved to:${NC} $PLOT_FILE"
fi
echo ""
