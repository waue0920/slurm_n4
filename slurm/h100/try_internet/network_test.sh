#!/bin/bash

# Script to test network connectivity and measure download speed
echo "Node: $(hostname)"

# Test basic connectivity to an external server
if curl -s --head http://www.google.com | grep '200 OK' > /dev/null; then
    echo "Internet connectivity: OK"
#    wget -q --show-progress https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.10.1.tar.xz
else
    echo "Internet connectivity: FAILED"
    exit 1
fi

