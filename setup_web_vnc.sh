#!/bin/bash
# setup_web_vnc.sh

echo "Setting up web VNC..."

# Install packages
sudo apt update
sudo apt install -y xvfb x11vnc fluxbox net-tools

# Start Xvfb
Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1

# Start window manager  
fluxbox &

# Start VNC
x11vnc -display :1 -nopw -forever -shared -rfbport 5900 &

# Install noVNC
git clone https://github.com/novnc/noVNC.git
cd noVNC
./utils/novnc_proxy --vnc localhost:5900 --listen 6080 &

echo "========================================"
echo "Web VNC is running!"
echo "Go to your Codespace ports tab:"
echo "1. Find port 6080"
echo "2. Make it public"
echo "3. Click the globe icon to open"
echo "========================================"