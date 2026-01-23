print("Script started")

import torch
import time
import torch.quantization

from en_es_translation.utils import load_model


# ----------------------------
# Helper function: benchmark
# ----------------------------
def benchmark(model, runs=20):
    """
    Measure inference time by running the model multiple times.
    Using a small number of runs because transformer models are slow on CPU.
    """
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(["Hello world"])
    end = time.time()
    return end - start


# ----------------------------
# Load model
# ----------------------------
print("Before load_model()")
model, device = load_model()
print("After load_model()")

# Force CPU (required for quantization)
model = model.to("cpu")
model.eval()
print("Model moved to CPU")

# ----------------------------
# Sanity check: inference works
# ----------------------------
with torch.no_grad():
    output = model(["Hello world"])
    print("FP32 output:", output)

# ----------------------------
# Benchmark FP32 model
# ----------------------------
fp32_time = benchmark(model)
print("FP32 inference time:", fp32_time)

# ----------------------------
# Apply dynamic quantization
# ----------------------------
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # Quantize Linear layers only
    dtype=torch.qint8
)

quantized_model.eval()
print("Model quantized")

# ----------------------------
# Sanity check: quantized inference
# ----------------------------
with torch.no_grad():
    q_output = quantized_model(["Hello world"])
    print("INT8 output:", q_output)

# ----------------------------
# Benchmark quantized model
# ----------------------------
int8_time = benchmark(quantized_model)
print("INT8 inference time:", int8_time)

# ----------------------------
# Save model sizes
# ----------------------------
torch.save(model.state_dict(), "model_fp32.pt")
torch.save(quantized_model.state_dict(), "model_int8.pt")

print("Models saved (model_fp32.pt, model_int8.pt)")
