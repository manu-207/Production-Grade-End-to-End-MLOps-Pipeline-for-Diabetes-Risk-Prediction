#!/bin/bash
set -e

echo "=============================="
echo " Installing Python 3.10"
echo "=============================="

sudo apt update -y
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

python3.10 --version

echo "=============================="
echo " Creating Python Virtual Env"
echo "=============================="

cd /home/ubuntu

if [ ! -d "mlflow-venv" ]; then
  python3.10 -m venv mlflow-venv
fi

source mlflow-venv/bin/activate

python --version

echo "=============================="
echo " Installing MLflow & Deps"
echo "=============================="

pip install --upgrade pip
pip install mlflow boto3 psycopg2-binary awscli

mlflow --version

echo "=============================="
echo " Creating MLflow systemd service"
echo "=============================="

sudo tee /etc/systemd/system/mlflow.service > /dev/null <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu

Environment="PATH=/home/ubuntu/mlflow-venv/bin:/usr/bin:/bin"

ExecStart=/home/ubuntu/mlflow-venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://manu7-mlops-mlflow-artifacts --host 0.0.0.0 --port 5000 --allowed-hosts * --cors-allowed-origins *

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "=============================="
echo " Reloading & Starting MLflow"
echo "=============================="

sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl restart mlflow

echo "=============================="
echo " Checking MLflow Status"
echo "=============================="

sudo systemctl status mlflow --no-pager || true

echo "=============================="
echo " MLflow Logs (Last 20 lines)"
echo "=============================="

sudo journalctl -u mlflow -n 20 --no-pager || true

echo "=============================="
echo " MLflow Installation Complete"
echo "=============================="
echo " Access UI: http://<EC2-IP>:5000"
