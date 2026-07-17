# LLM Hosting Stack

vLLM (model serving) + LiteLLM (API proxy with key management) on a single-GPU server.

Stack:
- **vLLM** — high-throughput inference engine serving Ornith-1.0-35B (or any OpenAI-compatible model)
- **LiteLLM** — API gateway with virtual keys, per-user budgets, rate limits, and spend tracking
- **PostgreSQL** — persists keys, users, teams, and usage data

## Pre-flight: Detect Your GPU and Drivers

SSH into the server and run these commands in order.

### 1. Check if NVIDIA drivers are already loaded

```bash
nvidia-smi
```

If this works, you'll see your GPU model, driver version, CUDA version, and VRAM. Example output for an RTX PRO 6000:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx    Driver Version: 550.xx    CUDA Version: 12.4          |
|-------------------------------+----------------------+----------------------+
| GPU Name                      |  Memory-Usage        |
|===============================+======================|
|   0 NVIDIA RTX PRO 6000       |   1234MiB / 98304MiB |
+-------------------------------+----------------------+
```

Key things to note:
- **Driver version** (550.xx or higher recommended for Blackwell)
- **CUDA Version** (12.x required)
- **Total VRAM** — should show ~96GB for an RTX PRO 6000
- GPU name — confirms which card you have

### 2. Identify the exact GPU

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

### 3. Check NVIDIA kernel module

```bash
lsmod | grep nvidia
```

Should show `nvidia`, `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset`.

### 4. Check CUDA toolkit version

```bash
nvcc --version
```

Not strictly required (vLLM ships its own CUDA libs), but useful to know.

### 5. Ubuntu: check recommended driver

```bash
ubuntu-drivers devices
```

This lists recommended proprietary driver packages. Look for a line like:
```
driver   : nvidia-driver-550 - distro non-free recommended
```

## Installing NVIDIA Drivers (if missing)

### Option A: Ubuntu's proprietary-driver repo (recommended)

```bash
# Identify the recommended driver
ubuntu-drivers devices

# Install it (replace 550 with the recommended version)
sudo apt update
sudo apt install -y nvidia-driver-550

# Install CUDA toolkit (optional, for nvcc)
sudo apt install -y nvidia-cuda-toolkit

# Reboot
sudo reboot
```

After reboot, verify with `nvidia-smi`.

### Option B: NVIDIA's .run installer (for custom kernels / edge cases)

```bash
# Check your kernel version and GCC
uname -r
gcc --version

# Download the driver for your GPU
# For RTX PRO 6000 (Blackwell): https://www.nvidia.com/download/driverResults.aspx/235794/
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/550.xx/NVIDIA-Linux-x86_64-550.xx.run

# Stop any X server / display manager first
sudo telinit 3

# Run installer
chmod +x NVIDIA-Linux-x86_64-*.run
sudo ./NVIDIA-Linux-x86_64-*.run

# Reboot
sudo reboot
```

### Driver version guide

| GPU Architecture | Minimum Driver | Recommended |
|---|---|---|
| RTX PRO 6000 (Blackwell) | 550.x | 550.x |
| RTX 4090 / A6000 (Ada) | 535.x | 545.x |
| A100 / A6000 (Ampere) | 470.x | 535.x |
| V100 (Volta) | 418.x | 470.x |

### Install NVIDIA Container Toolkit

Required for GPU access inside Docker containers.

```bash
# Add the NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify GPU access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If this shows your GPU, you're ready.

## Quick Start

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env with your secrets
nano .env
```

Set at minimum:
- `LITELLM_MASTER_KEY` — generate with `openssl rand -hex 16` and prefix with `sk-`
- `POSTGRES_PASSWORD` — generate with `openssl rand -hex 16`
- `HUGGING_FACE_HUB_TOKEN` — optional for Ornith (public model), set if you hit rate limits

### 2. Start the stack

```bash
docker compose up -d
```

This starts PostgreSQL, then LiteLLM, then vLLM. The first start downloads the model from Hugging Face (~70GB for Ornith-1.0-35B-FP8) which can take 10-30 minutes depending on your connection.

### 3. Watch the logs

```bash
# All services
docker compose logs -f

# Just vLLM (to watch model download progress)
docker compose logs -f vllm

# Just LiteLLM
docker compose logs -f litellm
```

vLLM is ready when you see:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

LiteLLM is ready when you see:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:4000
```

## Testing the Setup

### 1. Test vLLM directly (via docker exec)

```bash
# Ask vLLM directly (shortcut)
docker compose exec vllm curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith-1.0",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 20,
    "temperature": 0.6
  }'
```

If this returns a response with `choices`, vLLM is working.

### 2. Test LiteLLM proxy

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith-1.0",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 20,
    "temperature": 0.6
  }'
```

### 3. Generate a virtual key for a user

```bash
curl -s http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["ornith-1.0"],
    "metadata": {"user": "alice@example.com"},
    "max_budget": 50.0,
    "budget_duration": "30d"
  }'
```

Save the returned `key` value — this is Alice's personal API key.

### 4. Use the virtual key

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer <alices-virtual-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith-1.0",
    "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime."}],
    "max_tokens": 512,
    "temperature": 0.6
  }'
```

The response includes a `reasoning_content` field (the model's thinking trace) and `content` (the final answer).

### 5. Check spend

```bash
# Spend per key
curl -s http://localhost:4000/key/info?key=<virtual-key> \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"

# Spend per user
curl -s "http://localhost:4000/user/info?user_id=<user-id>" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

## Model Management

### Swapping models

Edit `.env` and change the `MODEL` variable:

```env
MODEL=deepreinforce-ai/Ornith-1.0-9B
```

Then recreate the vLLM container:

```bash
docker compose up -d vllm
```

vLLM will download the new model on next start.

### Common models and their flags

| Model | VRAM | Flags to add/change |
|---|---|---|
| Ornith-1.0-9B | ~19GB | Drop `--kv-cache-dtype fp8`, use `--max-model-len 32768` |
| Ornith-1.0-35B-FP8 | ~55GB | Current default |
| Ornith-1.0-397B-FP8 | ~200GB | Add `--tensor-parallel-size N` (multi-GPU) |
| Qwen3.5-72B | ~140GB FP16 | Use AWQ/FP8 quantization |
| Llama 3.1-70B | ~140GB FP16 | Set `HUGGING_FACE_HUB_TOKEN`, use AWQ |

To change flags, edit the `command` section under `vllm` in `docker-compose.yaml`.

### Multiple models (advanced)

You can run multiple vLLM instances on different ports for different models, each configured in LiteLLM as a separate model group:

```yaml
model_list:
  - model_name: ornith-1.0
    litellm_params:
      model: openai/ornith-1.0
      api_base: http://vllm-ornith:8000/v1
      api_key: EMPTY
  - model_name: ornith-9b
    litellm_params:
      model: openai/ornith-9b
      api_base: http://vllm-9b:8001/v1
      api_key: EMPTY
```

## Key Management Reference

| Action | Endpoint | Auth |
|---|---|---|
| Generate key | `POST /key/generate` | Master key |
| List keys | `GET /key/list` | Master key |
| Get key info | `GET /key/info?key=<key>` | Master key |
| Block key | `POST /key/block` | Master key |
| Unblock key | `POST /key/unblock` | Master key |
| Update key budget | `POST /key/update` | Master key |
| Create user | `POST /user/new` | Master key |
| User info | `GET /user/info?user_id=<id>` | Master key |
| Create team | `POST /team/new` | Master key |

Generate a key with a budget:
```bash
curl -s http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["ornith-1.0"],
    "max_budget": 25.0,
    "budget_duration": "30d",
    "max_parallel_requests": 5,
    "metadata": {"user": "bob@example.com", "team": "engineering"}
  }'
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client     │────▶│   LiteLLM    │────▶│     vLLM     │
│ (API key)    │     │  :4000       │     │  :8000       │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  PostgreSQL  │     │   GPU VRAM   │
                     │ (keys,spend) │     │ (model+KV)   │
                     └──────────────┘     └──────────────┘
```

- Clients authenticate via LiteLLM virtual keys
- LiteLLM validates keys, enforces budgets/limits, logs spend to PostgreSQL
- LiteLLM forwards requests to vLLM (internal, no public access)
- vLLM runs inference on the GPU

## Troubleshooting

### `nvidia-smi` not found → No driver installed
```bash
sudo apt install nvidia-driver-550 && sudo reboot
```

### `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` → NVIDIA Container Toolkit missing
```bash
sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

### CUDA out of memory / vLLM crashes on startup
The model + KV cache doesn't fit in VRAM. Try:
- Lower `--gpu-memory-utilization` (e.g., 0.80)
- Lower `--max-model-len` (e.g., 65536)
- Switch to a smaller model (e.g., Ornith-1.0-9B)
- Enable `--kv-cache-dtype fp8` (already set for the default config)

### Model download is slow or fails
- Set `HUGGING_FACE_HUB_TOKEN` in `.env` — unauthenticated downloads are rate-limited
- Check network connectivity: `curl -I https://huggingface.co`
- The model is ~70GB — this is normal for first download

### vLLM starts but returns 404
Make sure the model name in your request matches `--served-model-name` (default: `ornith-1.0`).

### LiteLLM returns 401 Unauthorized
You're using the wrong key. Use either:
- The master key (from `LITELLM_MASTER_KEY` in `.env`)
- A virtual key (generated via `/key/generate`)

### vLLM: `cudaErrorIllegalInstruction` during graph capture
The `--gpu-memory-utilization` is too high. Lower it to 0.80 and restart:
```bash
# Edit docker-compose.yaml, change 0.85 → 0.80
docker compose up -d vllm
```

### How to restart after changing config
```bash
docker compose down
docker compose up -d
```
