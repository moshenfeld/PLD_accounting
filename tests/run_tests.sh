#!/bin/bash
# Test runner script with common test scenarios

set -e  # Exit on error

echo "======================================"
echo "Random Allocation PLD Test Suite"
echo "======================================"
echo ""

# Parse arguments
COVERAGE=false
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE="-vv"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--coverage] [-v|--verbose]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest $VERBOSE"

if [ "$COVERAGE" = true ]; then
    echo "Running with coverage analysis..."
    PYTEST_CMD="$PYTEST_CMD --cov=../ --cov-report=html --cov-report=term-missing"
fi

# Run tests
echo "Test command: $PYTEST_CMD"
echo ""
$PYTEST_CMD

# Show coverage report location if generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "======================================"
    echo "Coverage report generated at: htmlcov/index.html"
    echo "======================================"
fi

echo ""
echo "======================================"
echo "All tests completed!"
echo "======================================"
