from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import OptimizationRequest
from app.services import builder
from app.services.framework_client import FrameworkClient
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("volpe-integration")

app = FastAPI(title="Volpe Integration Service")

FRAMEWORK_URL = os.getenv("VOLPE_FRAMEWORK_URL", "http://localhost:8000")
fw_client = FrameworkClient(FRAMEWORK_URL)

@app.get("/")
async def root():
    return {"message": "Volpe Integration Service Operational"}

@app.post("/submit")
async def submit_job(request: OptimizationRequest):
    """
    Receives a notebook, packages it, and forwards it to the Volpe Framework.
    """
    logger.info(f"Received submission for user: {request.user_id}, notebook: {request.notebook_id}")

    try:
        #  Build the tarball context
        tar_stream = builder.create_build_context(request.notebook, request.requirements, request.notebook_id)
        
        # Save local copy for debugging/verification
        build_dir = "build_artifacts"
        os.makedirs(build_dir, exist_ok=True)
        local_tar_path = os.path.join(build_dir, f"{request.notebook_id}.tar")
        
        with open(local_tar_path, "wb") as f:
            f.write(tar_stream.getvalue())
            
        logger.info(f"Saved local build artifact to: {local_tar_path}")
        
        # Reset stream position for the HTTP upload
        tar_stream.seek(0)
        
        # Define metadata for the framework
        # The framework expects problemID, memory, targetInstances
        # We derive problemID from notebook_id for now
        metadata = {
            "problemID": request.notebook_id,
            "memory": 512.0, # Default memory in MB
            "targetInstances": 1
        }

        # Upload to Framework
        upload_success = await fw_client.upload_problem(request.notebook_id, tar_stream, metadata)
        
        if not upload_success:
            # logger.error(f"Failed to upload problem: {e}")
            raise HTTPException(status_code=502, detail="Failed to upload problem to Volpe Framework")
        

        #  Start the problem
        start_success = await fw_client.start_problem(request.notebook_id)
        
        if not start_success:
            raise HTTPException(status_code=502, detail="Failed to start problem on Volpe Framework")

        return {
            "status": "success", 
            "message": "Job submitted and started successfully", 
            "problem_id": request.notebook_id
        }

    except Exception as e:
        logger.error(f"Error processing submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{problem_id}")
async def get_results(problem_id: str):
    """
    Streams results from the Volpe Framework back to the user via SSE.
    """
    async def event_generator():
        try:
            async for chunk in fw_client.stream_results(problem_id):
                yield chunk
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")