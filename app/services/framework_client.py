import httpx
import logging

logger = logging.getLogger(__name__)

class FrameworkClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    async def upload_problem(self, problem_id: str, tar_stream, metadata: dict):
        url = f"{self.base_url}/api/v1/problems/{problem_id}"
        
        # Prepare the multipart upload
        files = {'image': (f'{problem_id}.tar', tar_stream, 'application/x-tar')}
        
        data = {'metadata': metadata} 
        
        logger.info(f"Uploading problem {problem_id} to {url}")
        
        async with httpx.AsyncClient() as client:
            try:
                import json
                data_str = {'metadata': json.dumps(metadata)}
                
                response = await client.post(url, files=files, data=data_str, timeout=30.0)
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                logger.error(f"Failed to upload problem: {e}")
                return False

    async def start_problem(self, problem_id: str):
        url = f"{self.base_url}/api/v1/problems/{problem_id}/start"
        logger.info(f"Starting problem {problem_id}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, timeout=10.0)
                response.raise_for_status()
                return True
            except httpx.HTTPError as e:
                logger.error(f"Failed to start problem: {e}")
                return False

    async def stream_results(self, problem_id: str):
        url = f"{self.base_url}/api/v1/problems/{problem_id}/results"
        logger.info(f"Connecting to result stream for {problem_id}")
        
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', url, timeout=None) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk