#!/bin/bash

app_env=${1:-development}

# Prefer activating Python virtual environment if present
if [ -f "bin/activate" ]; then
    . bin/activate
fi

# Development environment commands
dev_commands() {
    echo "Running development environment commands..."
    python3 hello.py
}

# Production environment commands
prod_commands() {
    echo "Running production environment commands..."
    python3 hello.py
}

# Decide environment based on argument
if [ "$app_env" = "production" ] || [ "$app_env" = "prod" ] ; then
    echo "Production environment detected"
    prod_commands
else
    echo "Development environment detected"
    dev_commands
fi
