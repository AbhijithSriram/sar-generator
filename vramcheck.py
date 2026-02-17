# Verify Ollama memory usage
import subprocess

# Check Mistral 7B (our primary model)
result = subprocess.run(['ollama', 'show', 'mistral:7b'],
                       capture_output=True, text=True)
print("=== Mistral 7B Model Info ===")
print(result.stdout)

# Check running models and GPU usage
result2 = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
print("\n=== Currently Running Models ===")
print(result2.stdout if result2.stdout else "No models currently loaded")

# Mistral 7B at Q4_K_M quantization uses ~4.4GB VRAM
# RTX 4060 8GB = plenty of headroom for inference