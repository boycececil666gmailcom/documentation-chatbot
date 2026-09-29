#!/bin/bash

# Define log_step function for beautiful ANSI output
log_step() {
    echo -e "\n\033[1;96m========================================================\033[0m"
    echo -e "\033[1;92m>>> $1 [$(basename "$0")] $2\033[0m"
    echo -e "\033[1;96m========================================================\033[0m\n"
}


log_step "Building Local Docker Images"
docker info >/dev/null 2>&1 || { echo -e "\033[1;31mError: Docker daemon is not running. Please start Docker and try again.\033[0m"; exit 1; }

echo "Building core chatbot backend image..."
docker build -t documentation-chatbot-backend:latest -f Dockerfile . || exit 1


