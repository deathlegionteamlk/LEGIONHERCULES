#!/bin/bash
# LEGIONHERCULES One-Command Installation Script
# Usage: curl -fsSL https://legionhercules.dev/install.sh | bash
# Or:    wget -qO- https://legionhercules.dev/install.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/.legionhercules}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
REPO_URL="https://github.com/deathlegion/legionhercules"
VERSION="${VERSION:-latest}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██╗     ███████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗        ║
║   ██║     ██╔════╝██╔════╝ ██║██╔═══██╗████╗  ██║        ║
║   ██║     █████╗  ██║  ███╗██║██║   ██║██╔██╗ ██║        ║
║   ██║     ██╔══╝  ██║   ██║██║██║   ██║██║╚██╗██║        ║
║   ███████╗███████╗╚██████╔╝██║╚██████╔╝██║ ╚████║        ║
║   ╚══════╝╚══════╝ ╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═══╝        ║
║                                                           ║
║   ██╗  ██╗███████╗██████╗  ██████╗██╗   ██╗██╗     ███████╗███████╗ ║
║   ██║  ██║██╔════╝██╔══██╗██╔════╝██║   ██║██║     ██╔════╝██╔════╝ ║
║   ███████║█████╗  ██████╔╝██║     ██║   ██║██║     █████╗  ███████╗ ║
║   ██╔══██║██╔══╝  ██╔══██╗██║     ██║   ██║██║     ██╔══╝  ╚════██║ ║
║   ██║  ██║███████╗██║  ██║╚██████╗╚██████╔╝███████╗███████╗███████║ ║
║   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝ ║
║                                                           ║
║   Autonomous CLI Framework with Parallel Agent Execution  ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS=Linux;;
        Darwin*)    OS=Mac;;
        CYGWIN*|MINGW*|MSYS*) OS=Windows;;
        *)          OS=Unknown;;
    esac
    log_info "Detected OS: $OS"
}

# Check Python version
check_python() {
    log_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        log_error "Python is not installed. Please install Python 3.9 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Found Python $PYTHON_VERSION"
    
    # Check version >= 3.9
    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        log_error "Python 3.9 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Install Python if needed (Linux only)
install_python() {
    if [ "$OS" = "Linux" ]; then
        log_info "Installing Python..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-pip
        elif command -v pacman &> /dev/null; then
            sudo pacman -S python python-pip
        else
            log_warning "Could not install Python automatically. Please install manually."
        fi
    fi
}

# Create directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    mkdir -p "$HOME/.legionhercules/config"
    mkdir -p "$HOME/.legionhercules/checkpoints"
    mkdir -p "$HOME/.legionhercules/indexes"
}

# Install LEGIONHERCULES
install_legionhercules() {
    log_info "Installing LEGIONHERCULES..."
    
    cd "$INSTALL_DIR"
    
    # Clone or update repository
    if [ -d "$INSTALL_DIR/repo" ]; then
        log_info "Updating existing installation..."
        cd "$INSTALL_DIR/repo"
        git pull origin main || true
    else
        log_info "Cloning repository..."
        git clone "$REPO_URL" "$INSTALL_DIR/repo" || {
            log_warning "Git clone failed, trying alternative method..."
            # Fallback: download and install via pip
            pip3 install --user legionhercules || {
                log_error "Failed to install LEGIONHERCULES"
                exit 1
            }
            return
        }
    fi
    
    # Create virtual environment
    log_info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$INSTALL_DIR/venv"
    
    # Activate and install
    source "$INSTALL_DIR/venv/bin/activate"
    pip install --upgrade pip
    pip install "$INSTALL_DIR/repo"
    
    deactivate
}

# Create wrapper script
create_wrapper() {
    log_info "Creating wrapper script..."
    
    cat > "$BIN_DIR/legionhercules" << EOF
#!/bin/bash
# LEGIONHERCULES wrapper script

source "$INSTALL_DIR/venv/bin/activate"
exec legionhercules "\$@"
EOF
    
    chmod +x "$BIN_DIR/legionhercules"
    
    # Create lh alias
    ln -sf "$BIN_DIR/legionhercules" "$BIN_DIR/lh"
}

# Setup shell integration
setup_shell() {
    log_info "Setting up shell integration..."
    
    SHELL_RC=""
    case "$SHELL" in
        */bash) SHELL_RC="$HOME/.bashrc" ;;
        */zsh)  SHELL_RC="$HOME/.zshrc" ;;
        */fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
    esac
    
    if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
        if ! grep -q "$BIN_DIR" "$SHELL_RC"; then
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$SHELL_RC"
            log_info "Added $BIN_DIR to PATH in $SHELL_RC"
        fi
    fi
    
    # Create config file
    cat > "$HOME/.legionhercules/config/config.yaml" << 'EOF'
llm:
  provider: ollama
  model: llama3.2
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 4096

orchestrator:
  max_concurrent_tasks: 5
  task_timeout: 300.0
  retry_attempts: 3
  enable_parallel: true

logging:
  level: INFO
  use_rich: true

agents:
  - name: default
    description: Default general-purpose agent
    system_prompt: You are a helpful AI assistant.
    tools:
      - file_read
      - file_write
      - file_edit
      - bash
      - web_search
    max_iterations: 10
    timeout_seconds: 120.0
EOF
}

# Check Ollama
check_ollama() {
    log_info "Checking Ollama..."
    
    if command -v ollama &> /dev/null; then
        log_success "Ollama is installed"
        
        # Check if running
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            log_success "Ollama is running"
        else
            log_warning "Ollama is installed but not running"
            log_info "Start Ollama with: ollama serve"
        fi
    else
        log_warning "Ollama is not installed"
        log_info "Install Ollama from: https://ollama.com"
        log_info "For local LLM inference, Ollama is required"
    fi
}

# Print completion message
print_completion() {
    echo ""
    log_success "LEGIONHERCULES has been installed successfully!"
    echo ""
    echo -e "${GREEN}Installation Details:${NC}"
    echo "  Installation directory: $INSTALL_DIR"
    echo "  Binary location: $BIN_DIR/legionhercules"
    echo "  Config directory: $HOME/.legionhercules"
    echo ""
    echo -e "${GREEN}Quick Start:${NC}"
    echo "  1. Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    echo "  2. Start Ollama: ollama serve"
    echo "  3. Pull a model: ollama pull llama3.2"
    echo "  4. Start chatting: legionhercules chat"
    echo ""
    echo -e "${GREEN}Useful Commands:${NC}"
    echo "  legionhercules --help       Show help"
    echo "  legionhercules chat         Start interactive chat"
    echo "  legionhercules models       List available models"
    echo ""
    echo -e "${BLUE}Developed by Death Legion Team Coders Demo X HEXA${NC}"
}

# Uninstall function
uninstall() {
    log_info "Uninstalling LEGIONHERCULES..."
    
    rm -rf "$INSTALL_DIR"
    rm -f "$BIN_DIR/legionhercules"
    rm -f "$BIN_DIR/lh"
    
    log_success "LEGIONHERCULES has been uninstalled"
}

# Update function
update() {
    log_info "Updating LEGIONHERCULES..."
    install_legionhercules
    log_success "LEGIONHERCULES has been updated"
}

# Main function
main() {
    # Handle arguments
    case "${1:-}" in
        --uninstall)
            uninstall
            exit 0
            ;;
        --update)
            update
            exit 0
            ;;
        --help|-h)
            echo "LEGIONHERCULES Installer"
            echo ""
            echo "Usage:"
            echo "  curl -fsSL https://legionhercules.dev/install.sh | bash"
            echo "  wget -qO- https://legionhercules.dev/install.sh | bash"
            echo ""
            echo "Options:"
            echo "  --uninstall    Remove LEGIONHERCULES"
            echo "  --update       Update to latest version"
            echo "  --help         Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  INSTALL_DIR    Installation directory (default: ~/.legionhercules)"
            echo "  BIN_DIR        Binary directory (default: ~/.local/bin)"
            echo "  VERSION        Version to install (default: latest)"
            exit 0
            ;;
    esac
    
    print_banner
    detect_os
    check_python
    setup_directories
    install_legionhercules
    create_wrapper
    setup_shell
    check_ollama
    print_completion
}

# Run main function
main "$@"
