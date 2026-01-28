# Volpe Integration Service

This service acts as an integration layer for the Volpe Framework. It is designed to bridge the gap between external clients (such as an LLM controller or a user interface) and the core Volpe execution environment.

The service accepts code submissions (structured as Jupyter notebooks) and optional data files, packages them into self-contained Docker environments, and dispatches them to the Volpe Framework for optimization and execution. It also provides a streaming endpoint to relay real-time results back to the client.

## Dependent repos

- [VolPE Framework](https://github.com/aadit-n3rdy/volpe-framework/blob/main/master/api/controller.go#L37)
- [EvOC v2 LLM Microservice](https://github.com/Evolutionary-Algorithms-On-Click/evocv2_llm_microservice)

## Prerequisites

To run this service locally, ensure you have the following installed:

*   **VolPE Framework** master and worker up and running in port 8000 (WSL or Linux) - refer [VolPE Framework](https://github.com/aadit-n3rdy/volpe-framework/blob/main/master/api/controller.go#L37) dev docs (yet ot be built)
*   **Python 3.10** or higher.
*   **Docker Desktop** or **Docker Engine**: The service requires access to the Docker daemon to build and export images. Ensure Docker is running before starting the application.
*   **uv** (Recommended): This project uses `uv` for fast package management, though standard `pip` can also be used.

## Installation and Setup

1.  **Clone the repository**
    ``` bash
    git clone https://github.com/Evolutionary-Algorithms-On-Click/volpe-integration
    ```


2.  **Install dependencies**
    If using `uv`:
    ```bash
    uv sync
    ```
    Or using standard `pip`:
    ```bash
    pip install -r requirements.txt
    ```
## Running the Service

Start the FastAPI server using `fastapi` or `uvicorn`.

```bash
# Using fastapi CLI
fastapi run app/main.py --port 9000 --reload 

```

The service will be available at `http://localhost:9000`.

## API Documentation

### 1. Submit Job
**Endpoint:** `POST /submit`

This endpoint accepts a multipart/form-data request to submit a new optimization job. Builds the .tar file and sends to volpe framework

**Parameters:**

*   `request_data` (Form Field, Required): A JSON string representing the optimization request. It must validate against the `OptimizationRequest` schema. Typically a the same notebook structure returned from [EvOC v2 LLM Microservice](https://github.com/Evolutionary-Algorithms-On-Click/evocv2_llm_microservice)
    *   **Structure:**
        ```json
        {
          "user_id": "string",
          "notebook_id": "string",
          "notebook": {
            "cells": [ ... ],
            "requirements": "numpy\npandas"
          },
          "preferences": { ... }
        }
        ```
*   `file` (File Field, Optional): A binary file (e.g., `data.csv`) that will be uploaded and placed into the execution container's working directory. If a file named `data.csv` is uploaded, it will override any default data file.

**Response:**
Returns a JSON object containing the `status`, `message`, and the `problem_id`.

### 2. Stream Results
**Endpoint:** `GET /results/{problem_id}`

This endpoint opens a Server Sent Events (SSE) stream to relay updates from the Volpe Framework for a specific problem.

**Response:**
A text stream of events (standard SSE format).

### 3. Health Check
**Endpoint:** `GET /`

Returns a simple status message to verify the service is operational.

## Architecture and Workflow

When a request is received at `/submit`, the service performs the following steps:

1.  **Validation**: It parses the JSON `request_data` and validates it against the internal Pydantic models.
2.  **Context Creation**:
    *   It extracts code cells from the provided notebook.
    *   It wraps the user code using a standard template (`wrapper_template.py`) to ensure compatibility with the Volpe runtime.
    *   It prepares a tmp dir containing the generated `main.py`, a `Dockerfile`, `requirements.txt`, and necessary protobuf definitions.
    *   If a file was uploaded, it is written to this directory.
3.  **Image Build**: It uses the local Docker daemon to build a Docker image tagged with the notebook ID.
4.  **Export**: The built image is immediately exported as a tarball stream.
5.  **Dispatch**: The tarball and metadata are uploaded to the configured `VOLPE_FRAMEWORK_URL`.
6.  **Cleanup**: The Docker image is removed from the local daemon to save space, though the tarball is temporarily cached in `build_artifacts/` and the main.py in `debug_artifacts/` for debugging purposes.

