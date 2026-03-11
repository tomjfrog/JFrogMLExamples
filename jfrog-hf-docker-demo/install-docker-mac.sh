#!/usr/bin/env bash
#
# Install Homebrew (if missing) and Docker Desktop on macOS.
# Run in Terminal: ./install-docker-mac.sh
# You will be prompted for your macOS password when sudo is needed.
#
set -e

# --- 1. Install Homebrew if not present ---
if ! command -v brew &>/dev/null; then
  echo "Installing Homebrew (you may be prompted for your password)..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add Homebrew to PATH for Apple Silicon (M1/M2/M3)
  if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
else
  echo "Homebrew already installed."
  eval "$(brew shellenv)" 2>/dev/null || true
fi

# Ensure brew is on PATH
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
command -v brew || { echo "Homebrew not found on PATH. Add it and re-run."; exit 1; }

# --- 2. Install Docker Desktop ---
echo "Installing Docker Desktop..."
brew install --cask docker

echo ""
echo "Docker Desktop is installed."
echo ""
echo "Next steps (manual):"
echo "  1. Open Docker Desktop from Applications (or run: open -a Docker)"
echo "  2. Accept the terms if prompted and wait until Docker is running (whale icon in menu bar)."
echo "  3. In Terminal, run:"
echo "       cd \"$(cd "$(dirname "$0")" && pwd)\""
echo "       docker build -t jfrog-hf-demo:latest ."
echo "       docker run -p 8000:8000 jfrog-hf-demo:latest"
echo ""
