#!/bin/bash

# Simple LLM Inference Runner
# Edit the MODELS_TO_RUN list below to choose which models to run
# Usage: ./run_inference.sh [--run-full]

#===========================================
# CONFIGURATION - EDIT THIS LIST
#===========================================

# Wait time between status checks (in seconds)
WAIT_TIME=5

# Uncomment the models you want to run by removing the # at the beginning
MODELS_TO_RUN=(
    "gpt-4o"
)

# Maximum number of models to run simultaneously
MAX_CONCURRENT=8

#===========================================
# SCRIPT - DON'T EDIT BELOW THIS LINE
#===========================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Go to script directory
cd "$(dirname "$0")"

# Signal handler for clean exit
cleanup() {
    echo -e "\n${YELLOW}Received interrupt signal. Cleaning up...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "Terminating process $pid..."
            kill -TERM $pid 2>/dev/null || true
        fi
    done
    sleep 3
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "Force killing process $pid..."
            kill -KILL $pid 2>/dev/null || true
        fi
    done
    echo -e "${RED}Script interrupted. Some processes may have been terminated.${NC}"
    exit 130
}

trap cleanup SIGINT SIGTERM

# Create directories
mkdir -p ./logs
mkdir -p ./outputs

echo -e "${GREEN}=== LLM Inference Runner ===${NC}"
echo "Models to run: ${#MODELS_TO_RUN[@]}"
echo "Max concurrent: $MAX_CONCURRENT"
echo

# Show selected models
for model in "${MODELS_TO_RUN[@]}"; do
    echo -e "${YELLOW}- $model${NC}"
done
echo

# Track running processes
declare -a PIDS=()
declare -a RUNNING_MODELS=()

# Function to wait for available slot
wait_for_slot() {
    while [ ${#PIDS[@]} -ge $MAX_CONCURRENT ]; do
        echo "Currently running ${#PIDS[@]} processes (max: $MAX_CONCURRENT). Checking for completed processes..."

        # Check finished processes
        local new_pids=()
        local new_models=()
        local completed_count=0

        for i in "${!PIDS[@]}"; do
            local pid=${PIDS[$i]}
            local model=${RUNNING_MODELS[$i]}

            if kill -0 $pid 2>/dev/null; then
                new_pids+=($pid)
                new_models+=("$model")
            else
                wait $pid
                local exit_code=$?
                ((completed_count++))
                if [ $exit_code -eq 0 ]; then
                    echo -e "${GREEN}✓ Completed: $model${NC}"
                else
                    echo -e "${RED}✗ Failed: $model (exit code: $exit_code)${NC}"
                fi
            fi
        done

        PIDS=("${new_pids[@]}")
        RUNNING_MODELS=("${new_models[@]}")

        if [ $completed_count -gt 0 ]; then
            echo "Cleaned up $completed_count completed processes. Currently running: ${#PIDS[@]}"
        fi

        if [ ${#PIDS[@]} -ge $MAX_CONCURRENT ]; then
            echo "Still at max capacity. Waiting 2 seconds before next check..."
            sleep 2
        fi
    done

    echo "Available slots: $((MAX_CONCURRENT - ${#PIDS[@]}))"
}

# Start each model
echo "Starting inference..."
for model in "${MODELS_TO_RUN[@]}"; do
    wait_for_slot

    echo -e "${YELLOW}Starting: $model${NC}"
    if [ "$RUN_FULL" = true ]; then
        python3 ./main.py --model_name "$model" --run-full &
    else
        python3 ./main.py --model_name "$model" &
    fi

    PIDS+=($!)
    RUNNING_MODELS+=("$model")
    sleep 1
done

# Wait for all to complete
echo "Waiting for all processes to complete..."
echo "This may take a long time as models perform inference..."
echo "Press Ctrl+C to interrupt and clean up gracefully."
echo

while [ ${#PIDS[@]} -gt 0 ]; do
    echo "$(date): Checking status of ${#PIDS[@]} remaining processes..."

    # Check if any processes are still running
    new_pids=()
    new_models=()
    any_running=false

    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        model=${RUNNING_MODELS[$i]}

        if kill -0 $pid 2>/dev/null; then
            new_pids+=($pid)
            new_models+=("$model")
            any_running=true
            echo "  - Still running: $model (PID: $pid)"
        else
            wait $pid
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo -e "${GREEN}✓ Completed: $model${NC}"
            else
                echo -e "${RED}✗ Failed: $model (exit code: $exit_code)${NC}"
            fi
        fi
    done

    PIDS=("${new_pids[@]}")
    RUNNING_MODELS=("${new_models[@]}")

    # If no processes are running, break out
    if [ "$any_running" = false ]; then
        echo "All processes have completed!"
        break
    fi

    sleep $WAIT_TIME
done

echo -e "${GREEN}All models completed!${NC}"
echo "Check outputs in: ./outputs/"
echo "Check logs in: ./logs/"
