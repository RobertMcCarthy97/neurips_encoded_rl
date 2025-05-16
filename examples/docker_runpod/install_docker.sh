#!/bin/bash

# Configuration
USERNAME=$(whoami)  # Current user, change if needed
DOCKER_COMPOSE_VERSION="latest"  # Change if you need a specific version
ENABLE_DOCKER_EXPERIMENTAL=false  # Set to true if you want experimental features
VERIFY_INSTALL=true  # Set to false to skip verification steps

# Function to check if command succeeded
check_status() {
    if [ $? -ne 0 ] && [ "$2" != "ignore" ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Function to run command with sudo
run_sudo() {
    echo "Running: $1"
    sudo $1
    check_status "$1 failed" "$2"
}

# Print configuration
echo "Installing Docker with following configuration:"
echo "Username: $USERNAME"
echo "Docker Compose Version: $DOCKER_COMPOSE_VERSION"
echo "Enable Experimental Features: $ENABLE_DOCKER_EXPERIMENTAL"
echo "Verify Installation: $VERIFY_INSTALL"
echo

# Confirm with user
read -p "Continue with these settings? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborting..."
    exit 1
fi

# Remove old versions (ignore errors here since packages might not exist)
echo "Removing old Docker versions if they exist..."
run_sudo "apt-get remove -y docker docker-engine docker.io containerd runc" "ignore"

# Update package list
echo "Updating package list..."
run_sudo "apt-get update"

# Install prerequisites
echo "Installing prerequisites..."
run_sudo "apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release"

# Add Docker's official GPG key
echo "Adding Docker's GPG key..."
run_sudo "mkdir -p /etc/apt/keyrings"
# Fixed GPG key installation
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
check_status "Failed to add Docker's GPG key"

# Set up Docker repository
echo "Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
check_status "Failed to set up Docker repository"

# Update apt package index
echo "Updating package index..."
run_sudo "apt-get update"

# Install Docker Engine
echo "Installing Docker Engine..."
run_sudo "apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin"

# Add user to docker group
echo "Adding user to docker group..."
sudo usermod -aG docker "$USERNAME"
check_status "Failed to add user to docker group"

# Set experimental features if enabled
if [ "$ENABLE_DOCKER_EXPERIMENTAL" = true ] ; then
    echo "Enabling experimental features..."
    sudo mkdir -p /etc/docker
    echo '{"experimental": true}' | sudo tee /etc/docker/daemon.json
    run_sudo "systemctl restart docker"
fi

# Fix permissions
echo "Setting permissions..."
run_sudo "chmod 666 /var/run/docker.sock"

# Verify installation if enabled
if [ "$VERIFY_INSTALL" = true ] ; then
    echo "Verifying Docker installation..."
    docker --version
    check_status "Docker installation verification failed"
    
    echo "Running test container..."
    docker run hello-world
    check_status "Docker test container failed"
fi

echo
echo "Docker installation completed successfully!"
echo "NOTE: You may need to log out and back in for group changes to take effect"
echo "To verify after logging back in, run: docker run hello-world"