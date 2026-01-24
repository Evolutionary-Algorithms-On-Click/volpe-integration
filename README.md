# volpe-integration (as part of EvOCv2)

This module receives the LLM generated code cells from the controller_v2 microservice, packages the code files as docker image .tar file, sends to volpe system and send best individuals (with fitness) via SSE