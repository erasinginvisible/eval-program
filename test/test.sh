#!/bin/bash

# Function to print usage information
print_usage() {
    echo "Usage: $0 [--beige | --black] [-v|--verbose]"
    echo "  --beige    Run beige evaluation"
    echo "  --black    Run black evaluation"
    echo "  -v, --verbose    Enable verbose output"
}

# Initialize variables
eval_type=""
verbose=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --beige)
            eval_type="beige"
            shift
            ;;
        --black)
            eval_type="black"
            shift
            ;;
        -v|--verbose)
            verbose=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if eval_type is set
if [ -z "$eval_type" ]; then
    echo "Error: You must specify either --beige or --black"
    print_usage
    exit 1
fi

# Function to run evaluation
run_evaluation() {
    local type=$1
    echo "Running $type evaluation..."
    
    if [ "$verbose" = true ]; then
        echo "Removing old directories..."
    fi
    rm -rf "test/$type/proc"
    rm -rf "test/$type/out"
    
    if [ "$verbose" = true ]; then
        echo "Creating output directory..."
    fi
    mkdir -p "test/$type/out"
    
    if [ "$verbose" = true ]; then
        echo "Running Python script..."
    fi
    python3 "$type.py" -i "test/$type/res" -o "test/$type/out"
    
    echo "Evaluation complete. Results:"
    cat "test/$type/out/scores.json"
}

# Run the appropriate evaluation
run_evaluation "$eval_type"