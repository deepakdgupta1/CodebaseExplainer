# Setup Custom Qwen 2.5 Coder Model

This project is configured to use a custom quantized version of Qwen 2.5 Coder 32B. The model files are located in a common directory `~/ai-models` so they can be shared across projects.

## 1. Model Location

The model file and Modelfile have been moved to:
- `~/ai-models/Qwen2.5-Coder-32B-Instruct.IQ4_XS.gguf`
- `~/ai-models/Modelfile`

If you need to download it again in the future:
```bash
mkdir -p ~/ai-models
cd ~/ai-models
wget https://huggingface.co/CISCai/Qwen2.5-Coder-32B-Instruct-SOTA-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct.IQ4_XS.gguf
```

## 2. Update Ollama (Critical)

To support the 8-bit KV cache quantization, you must update Ollama to the latest version:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 3. Create the Ollama Model

Run the following command to create (or update) the custom model in Ollama from the common directory:

```bash
cd ~/ai-models
ollama create qwen2.5-coder-32b-custom -f Modelfile
```

## 4. Verify Setup

You can verify the model is working by running a simple prompt:

```bash
ollama run qwen2.5-coder-32b-custom "Hello, are you ready to analyze code?"
```

## 5. Run Analysis

Now you can run the CodebaseExplainer analysis as usual from the project directory:

```bash
cd /home/deeog/Desktop/CodebaseExplainer
./venv/bin/codehierarchy analyze /tmp/claude-dementia --output ./output
```

## Troubleshooting

### Memory Errors
If you see an error like `model requires more system memory (17.1 GiB) than is available`, it means your system does not have enough free RAM to load the model.
- Try closing other applications to free up RAM.
- If you cannot free enough memory, you may need to use a smaller quantized version of the model (e.g., Q3_K_M or Q4_0) or a smaller model parameter size (e.g., 14B or 7B).

### KV Cache Parameter
The `kv_cache_type` parameter was requested but is not currently supported by the installed version of Ollama in the `Modelfile`. The model uses the default KV cache settings.
