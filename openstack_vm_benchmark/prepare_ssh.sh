#!/bin/bash

# Configuration
NODES=("vm201" "vm202")
USER="ubuntu"

echo "[SSH] Starting SSH key generation and distribution..."

# 1. Generate SSH key if not exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "[SSH] Generating new RSA key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
else
    echo "[SSH] RSA key already exists."
fi

# 2. Add local key to authorized_keys
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh

# 3. Add nodes to known_hosts to avoid prompts
for node in "${NODES[@]}"; do
    ssh-keyscan -H $node >> ~/.ssh/known_hosts 2>/dev/null
done

echo "[SSH] Local setup complete. Attempting to sync with other nodes..."
echo "[SSH] Note: If this is the first time, you might be asked for a password for the other node."

# 4. Distribute key and authorized_keys to other nodes
# We use a loop to ensure all nodes have the same authorized_keys
PUB_KEY=$(cat ~/.ssh/id_rsa.pub)

for node in "${NODES[@]}"; do
    if [[ "$(hostname)" == *"$node"* ]]; then
        continue
    fi
    
    echo "[SSH] Syncing with $node..."
    # Copy the pub key to the remote authorized_keys
    ssh -o BatchMode=no $USER@$node "mkdir -p ~/.ssh && echo '$PUB_KEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    
    # Also copy the private key if we want to be able to jump FROM the other node back to here (Bi-directional)
    # The user asked for "雙向免密登入"
    scp ~/.ssh/id_rsa $USER@$node:~/.ssh/id_rsa
    scp ~/.ssh/id_rsa.pub $USER@$node:~/.ssh/id_rsa.pub
    ssh $USER@$node "chmod 600 ~/.ssh/id_rsa"
done

echo "[SSH] Bidirectional SSH setup attempt finished."
