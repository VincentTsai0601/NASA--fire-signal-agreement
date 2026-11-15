# Docker deployment

From the project directory, start the dashboard and Qwen together:

```bash
docker compose up --build
```

After the analyst is healthy, open http://localhost:8501. Later starts can use
`docker compose up` (add `-d` for background operation). Compose builds the
dashboard image automatically on the first run; use `--build` after source changes.

## Prerequisites

- Docker Engine with Compose 2.30 or newer.
- A supported NVIDIA GPU, working host NVIDIA driver, and NVIDIA Container Toolkit
  configured for Docker. These are needed for the vLLM analyst container.
- The existing `data/fire.sqlite`, readable by container UID 10001. Raw FIRMS
  archives and the database are kept outside the images.
- Internet access and disk space for the Python/vLLM images and initial Qwen
  download. The first launch can take several minutes.

Stop an existing host-side Qwen server before startup if it occupies GPU memory.
If another dashboard already uses port 8501, run
`DASHBOARD_PORT=8502 docker compose up` and open http://localhost:8502.
This workspace has an ignored local `.env` with `DASHBOARD_PORT=18501`, preserving
the existing host dashboard on 8501. Its Docker dashboard is at http://localhost:18501.

The default analyst is `Qwen/Qwen2.5-1.5B-Instruct`, using half precision, a
4096-token context and GPU memory utilization 0.85. The vLLM image defaults to
`vllm/vllm-openai:v0.14.1`; override with `VLLM_IMAGE` for a version compatible
with your host driver. Container validation of this version must be distinguished
from the earlier host-side Qwen benchmark.

Compose connects the dashboard to `http://analyst:8000/v1`. Only the dashboard
port is published, bound to localhost. You do not need a separate host-side vLLM
process or Python environment. The dashboard waits for the analyst health check.

## Persistent data and shutdown

The database is a read-only bind mount from `data/fire.sqlite`. The model download
is stored in the Compose-managed `model-cache` volume. Both survive:

```bash
docker compose down
```

`docker compose down -v` deletes the model cache; avoid it if you want to retain
downloaded weights. Keep a separate backup of `data/fire.sqlite` and your archives:
Docker images do not contain these data files.

## Rebuild a missing database using Docker

Stop the dashboard before replacing its database. With both standard archive
CSVs in the project directory, build and run the existing pipeline:

```bash
docker compose build dashboard
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/input,readonly" \
  --mount "type=bind,src=$PWD/data,dst=/app/data" \
  fire-signal-agreement:local python pipeline.py \
  --modis /input/fire_archive_M-C61_848110.csv \
  --viirs /input/fire_archive_SV-C2_848111.csv
docker compose up
```

Create `data/` first if it does not exist. Archive filenames can be changed in
the command. The pipeline writes the database to the host folder and prints its
build summary to the terminal.
A missing database produces a bind-mount error instead of silently creating a
directory or launching an empty dashboard.

## Verify the running deployment

```bash
docker compose ps
docker compose logs --tail=100 analyst dashboard
docker compose exec dashboard python smoke_test.py --output /tmp/docker-smoke.json
docker compose cp dashboard:/tmp/docker-smoke.json ./data/docker-smoke.json
```

The smoke test must exit zero before claiming the containerized analyst passes
the existing benchmark. For dashboard-only use without a GPU or model service:

```bash
docker compose up --build --no-deps dashboard
```

Charts and evidence remain available; the analyst will be unavailable.

## Export the dashboard image

```bash
docker image save -o /tmp/fire-signal-agreement.tar fire-signal-agreement:local
docker image load -i /tmp/fire-signal-agreement.tar
```

Copy the archive, `compose.yaml`, and `data/fire.sqlite` to the destination
computer. The destination still needs the vLLM image and model weights, downloaded
on its first start. Use `docker compose up --no-build` with the loaded dashboard
image. This export is the dashboard image, not a backup of the full deployment.

References: [vLLM Docker deployment](https://docs.vllm.ai/en/v0.14.1/deployment/docker/)
and [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/).

## Verification status

Initial diagnosis: the RTX 3090 was visible to host `nvidia-smi`, but NVIDIA
Container Toolkit was missing despite an existing Docker runtime entry. The user
installed and configured the toolkit; a CUDA container now detects the RTX 3090
successfully. On this Ubuntu host, the NVIDIA package repository is configured.
For the same initial issue on another machine, the repair is:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

The Docker restart interrupts running containers. Installation requires local sudo
authentication; do not share passwords in chat. Follow the
[NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
if the package repository is absent on another machine.

The updated dashboard image was built and tested against the existing SQLite
database. Container rendering verified the calibrated XGBoost calendar, year
selection, shared color scale and observed 2025 totals. Both Compose services
reached healthy status with Qwen running on the RTX 3090. All 102 host application
tests passed. The containerized analyst benchmark passed 5/5, all three demo cases
passed, and all four application scope checks passed. Evidence is saved in
`data/docker-smoke.json`.

The running dashboard's Ask button matched all three live smoke answers in
Streamlit AppTest, saved in `data/docker-dashboard-check.json`. This verifies
the real application with a live model inside Docker; it is not an independently
operated browser or a walkthrough recording. Existing host-side Qwen weights
were copied into the named model cache volume, so no additional model download
was needed for this deployment.

Persistence and startup were also verified by stopping both services and running
`docker compose up -d --wait`. Both returned healthy, live Qwen connectivity passed,
and all 155 calibrated weeks and 51 held-out evaluation weeks remained available.
