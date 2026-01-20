#!/bin/bash

URL="http://localhost:8000/translate"
REQUESTS=1000

echo "Running $REQUESTS requests..."

START=$(date +%s)

for i in $(seq 1 $REQUESTS); do
  curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello!"}' > /dev/null
done

END=$(date +%s)
ELAPSED=$((END - START))

echo "Done!"
echo "Total time: ${ELAPSED}s"
echo "Average time per request: $(echo "scale=3; $ELAPSED/$REQUESTS" | bc)s"
