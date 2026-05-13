#!/bin/bash
ALB="http://manu7-mlops-alb-1399317811.ap-south-1.elb.amazonaws.com/predict"

echo "Sending 200 predictions..."

for i in $(seq 1 200); do
  # Randomize inputs to get varied data
  PREGNANCIES=$((RANDOM % 10 + 1))
  GLUCOSE=$((RANDOM % 150 + 70))        # 70–220
  BP=$((RANDOM % 50 + 50))              # 50–100
  SKIN=$((RANDOM % 40 + 10))            # 10–50
  INSULIN=$((RANDOM % 200 + 20))        # 20–220
  BMI=$(echo "$((RANDOM % 200 + 150))" | awk '{printf "%.1f", $1/10}')  # 15.0–35.0
  DPF=$(echo "$((RANDOM % 20 + 1))" | awk '{printf "%.2f", $1/10}')     # 0.1–2.0
  AGE=$((RANDOM % 50 + 20))             # 20–70

  curl -s -X POST "$ALB" \
    -H "Content-Type: application/json" \
    -d "{
      \"Pregnancies\": $PREGNANCIES,
      \"Glucose\": $GLUCOSE,
      \"BloodPressure\": $BP,
      \"SkinThickness\": $SKIN,
      \"Insulin\": $INSULIN,
      \"BMI\": $BMI,
      \"DiabetesPedigreeFunction\": $DPF,
      \"Age\": $AGE
    }" > /dev/null

  echo "[$i/200] Glucose=$GLUCOSE BMI=$BMI Age=$AGE"
  sleep 0.5   # 0.5s gap → ~2 req/sec, creates a nice time-series graph
done

echo "Done! Refresh Grafana."
