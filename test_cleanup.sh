#!/bin/bash

# Test script for verifying resource cleanup after fixes
# Tests interruption handling and GPU memory release

echo "========================================="
echo "   HIL-SERL Resource Cleanup Test       "
echo "========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check GPU memory
check_gpu() {
    echo -e "${YELLOW}GPU Memory Status:${NC}"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
}

# Function to kill process after delay
kill_after_delay() {
    local pid=$1
    local delay=$2
    local name=$3
    sleep $delay
    echo -e "\n${YELLOW}Sending SIGINT to $name (PID: $pid)${NC}"
    kill -INT $pid 2>/dev/null
}

# Test 1: record_success_fail.py
test_record_success_fail() {
    echo -e "\n${GREEN}Test 1: Testing record_success_fail.py cleanup${NC}"
    check_gpu
    
    # Start the process in background
    python /home/hanyu/code/hil-serl/examples/record_success_fail.py \
        --exp_name hirol_unifined \
        --successes_needed 5 &
    local pid=$!
    
    # Kill after 5 seconds
    kill_after_delay $pid 5 "record_success_fail.py" &
    
    # Wait for process to exit
    wait $pid 2>/dev/null
    local exit_code=$?
    
    sleep 2
    echo -e "${YELLOW}After cleanup:${NC}"
    check_gpu
    
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${RED}FAILED: Process still running!${NC}"
        kill -9 $pid 2>/dev/null
        return 1
    else
        echo -e "${GREEN}PASSED: Process cleaned up properly${NC}"
        return 0
    fi
}

# Test 2: record_demos.py
test_record_demos() {
    echo -e "\n${GREEN}Test 2: Testing record_demos.py cleanup${NC}"
    check_gpu
    
    python /home/hanyu/code/hil-serl/examples/record_demos.py \
        --exp_name hirol_unifined \
        --successes_needed 2 &
    local pid=$!
    
    kill_after_delay $pid 5 "record_demos.py" &
    
    wait $pid 2>/dev/null
    local exit_code=$?
    
    sleep 2
    echo -e "${YELLOW}After cleanup:${NC}"
    check_gpu
    
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${RED}FAILED: Process still running!${NC}"
        kill -9 $pid 2>/dev/null
        return 1
    else
        echo -e "${GREEN}PASSED: Process cleaned up properly${NC}"
        return 0
    fi
}

# Test 3: train_rlpd.py actor
test_train_actor() {
    echo -e "\n${GREEN}Test 3: Testing train_rlpd.py actor cleanup${NC}"
    check_gpu
    
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
    
    python /home/hanyu/code/hil-serl/examples/train_rlpd.py \
        --exp_name hirol_unifined \
        --actor \
        --checkpoint_path /tmp/test_checkpoint &
    local pid=$!
    
    kill_after_delay $pid 10 "train_rlpd.py actor" &
    
    wait $pid 2>/dev/null
    local exit_code=$?
    
    sleep 2
    echo -e "${YELLOW}After cleanup:${NC}"
    check_gpu
    
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${RED}FAILED: Process still running!${NC}"
        kill -9 $pid 2>/dev/null
        return 1
    else
        echo -e "${GREEN}PASSED: Process cleaned up properly${NC}"
        return 0
    fi
}

# Main test execution
main() {
    local failed=0
    
    echo -e "${YELLOW}Starting cleanup tests...${NC}\n"
    echo "Initial GPU status:"
    check_gpu
    
    # Run tests based on argument
    case "$1" in
        1)
            test_record_success_fail || ((failed++))
            ;;
        2)
            test_record_demos || ((failed++))
            ;;
        3)
            test_train_actor || ((failed++))
            ;;
        all|"")
            test_record_success_fail || ((failed++))
            echo -e "\n${YELLOW}Waiting 5 seconds before next test...${NC}"
            sleep 5
            
            test_record_demos || ((failed++))
            echo -e "\n${YELLOW}Waiting 5 seconds before next test...${NC}"
            sleep 5
            
            test_train_actor || ((failed++))
            ;;
        *)
            echo "Usage: $0 [1|2|3|all]"
            echo "  1: Test record_success_fail.py"
            echo "  2: Test record_demos.py"
            echo "  3: Test train_rlpd.py actor"
            echo "  all: Run all tests (default)"
            exit 1
            ;;
    esac
    
    echo -e "\n========================================="
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All tests PASSED!${NC}"
    else
        echo -e "${RED}$failed test(s) FAILED!${NC}"
    fi
    echo "========================================="
    
    return $failed
}

# Run main function
main "$@"