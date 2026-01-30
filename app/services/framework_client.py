import httpx
import logging
import json


logger = logging.getLogger(__name__)

class FrameworkClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    async def upload_problem(self, problem_id: str, tar_stream, metadata: dict):
        url = f"{self.base_url}/problems/{problem_id}"
        
        # rewind stream
        if hasattr(tar_stream, 'seek'):
            tar_stream.seek(0)
            
        # verify not empty
        content_sample = tar_stream.read(10)
        if len(content_sample) == 0:
            logger.error("CRITICAL: Attempting to upload EMPTY file stream!")
            return False
        tar_stream.seek(0) # Rewind back after checking
        
        files = {'image': (f'{problem_id}.tar', tar_stream, 'application/x-tar')}
        
        data = {'metadata': json.dumps(metadata)}
        
        logger.info(f"Uploading problem {problem_id} to {url}")
        
        async with httpx.AsyncClient() as client:
            try:
                # httpx automatically merges 'files' and 'data' into multipart/form-data
                response = await client.post(url, files=files, data=data, timeout=30.0)
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                logger.error(f"Failed to upload problem: {e}")
                return False

    async def start_problem(self, problem_id: str):
        url = f"{self.base_url}/problems/{problem_id}/start"
        logger.info(f"Starting problem {problem_id}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(url, timeout=10.0)
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                logger.error(f"Failed to start problem: {e}")
                return False

    async def stream_results(self, problem_id: str):
        url = f"{self.base_url}/problems/{problem_id}/results"
        logger.info(f"Connecting to result stream for {problem_id}")
        
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', url, timeout=None) as response:
                async for chunk in response.aiter_bytes():
                    # print(chunk)
                    yield chunk