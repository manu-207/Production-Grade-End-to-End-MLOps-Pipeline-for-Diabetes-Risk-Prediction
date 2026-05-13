#!/bin/bash
set -e

echo "============================================"
echo "  Prometheus + Grafana Installation Script  "
echo "============================================"

# ─────────────────────────────────────────────
# VARIABLES — change these if needed
# ─────────────────────────────────────────────
PROM_VERSION="2.51.0"
PROM_DIR="/opt/prometheus"
ALB_TARGET="manu7-mlops-alb-149346731.ap-south-1.elb.amazonaws.com"

# ─────────────────────────────────────────────
# STEP 1 — Download & Install Prometheus
# ─────────────────────────────────────────────
echo ""
echo ">>> [1/6] Downloading Prometheus v${PROM_VERSION}..."
wget -q --show-progress \
  https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz

echo ">>> Extracting..."
tar -xvf prometheus-${PROM_VERSION}.linux-amd64.tar.gz

echo ">>> Moving to ${PROM_DIR}..."
sudo mv prometheus-${PROM_VERSION}.linux-amd64 ${PROM_DIR}

# ─────────────────────────────────────────────
# STEP 2 — Write Prometheus Config
# ─────────────────────────────────────────────
echo ""
echo ">>> [2/6] Writing prometheus.yml..."
sudo tee ${PROM_DIR}/prometheus.yml > /dev/null <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "diabetes-flask-app"
    static_configs:
      - targets: ["${ALB_TARGET}"]
    metrics_path: /metrics
    scheme: http
EOF
echo "    Config written to ${PROM_DIR}/prometheus.yml"

# ─────────────────────────────────────────────
# STEP 3 — Create Prometheus systemd Service
# ─────────────────────────────────────────────
echo ""
echo ">>> [3/6] Creating Prometheus systemd service..."
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus Monitoring
After=network.target

[Service]
ExecStart=${PROM_DIR}/prometheus \\
  --config.file=${PROM_DIR}/prometheus.yml \\
  --storage.tsdb.path=${PROM_DIR}/data
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus

echo "    Prometheus service status:"
sudo systemctl status prometheus --no-pager

# ─────────────────────────────────────────────
# STEP 4 — Install Grafana
# ─────────────────────────────────────────────
echo ""
echo ">>> [4/6] Installing Grafana prerequisites..."
sudo apt-get install -y apt-transport-https software-properties-common

echo ">>> Adding Grafana GPG key and repo..."
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" \
  | sudo tee /etc/apt/sources.list.d/grafana.list

echo ">>> Installing Grafana..."
sudo apt-get update -y && sudo apt-get install -y grafana

# ─────────────────────────────────────────────
# STEP 5 — Start Grafana
# ─────────────────────────────────────────────
echo ""
echo ">>> [5/6] Starting Grafana service..."
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

echo "    Grafana service status:"
sudo systemctl status grafana-server --no-pager

# ─────────────────────────────────────────────
# STEP 6 — Final Summary
# ─────────────────────────────────────────────
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<your-ec2-ip>")

echo ""
echo "============================================"
echo "           ✅ Installation Complete!        "
echo "============================================"
echo ""
echo "  Prometheus UI  →  http://${EC2_IP}:9090"
echo "  Grafana UI     →  http://${EC2_IP}:3000"
echo ""
echo "  Grafana default login:"
echo "    Username : admin"
echo "    Password : admin"
echo ""
echo "  Prometheus scraping:"
echo "    Target   : ${ALB_TARGET}"
echo "    Path     : /metrics"
echo "    Interval : 15s"
echo ""
echo "  Next steps:"
echo "    1. Open Grafana → Add Prometheus data source"
echo "       URL: http://localhost:9090"
echo "    2. Import a dashboard (ID: 11074 for Flask apps)"
echo "    3. Open EC2 Security Group ports 9090 and 3000"
echo "============================================"
