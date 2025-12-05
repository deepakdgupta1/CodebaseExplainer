# Setup Custom Qwen 2.5 Coder Model

This project is configured to use a custom quantized version of Qwen 2.5 Coder 32B. You must manually download the model file and create the Ollama model before running the analysis.

## 1. Download the Model

Download the `Qwen2.5-Coder-32B-Instruct.IQ4_XS.gguf` file (~19GB) from Hugging Face to the project root directory:

```bash
wget https://huggingface.co/CISCai/Qwen2.5-Coder-32B-Instruct-SOTA-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct.IQ4_XS.gguf
```

*Note: This is a large download. Ensure you have sufficient disk space and a stable connection.*

## 2. Create the Ollama Model

Once the download is complete, run the following command to create the custom model in Ollama:

```bash
ollama create qwen2.5-coder-32b-custom -f Modelfile
```

## 3. Verify Setup

You can verify the model is working by running a simple prompt:

```bash
ollama run qwen2.5-coder-32b-custom "Hello, are you ready to analyze code?"
```

## 4. Run Analysis

Now you can run the CodebaseExplainer analysis as usual:

```bash
./venv/bin/codehierarchy analyze /tmp/claude-dementia --output ./output
```
