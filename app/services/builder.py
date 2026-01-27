import io
import os
import tarfile
import shutil
import tempfile
import docker
from pathlib import Path
from app.models.schemas import Notebook

# Base paths
VOLPE_PY_PATH = Path("../volpe-py")
RESOURCES_PATH = Path("app/resources")

def extract_code_from_notebook(notebook: Notebook) -> str:
    code_blocks = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if cell.cell_name != 'results_and_plots' and cell.cell_name != 'evolution_loop':
                code_blocks.append(cell.source)
    return "\n\n".join(code_blocks)

def generate_wrapper_code(user_code: str) -> str:
    template_path = RESOURCES_PATH / "wrapper_template.py"
    if not template_path.exists():
        template_path = RESOURCES_PATH / "wrapper_main.py"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Wrapper template not found at {template_path}")

    with open(template_path, "r") as f:
        template = f.read()
    
    return template.replace("# {{USER_CODE_INJECTION}}", user_code)

def create_build_context_folder(tmp_dir: str, notebook: Notebook, requirements: str | None, notebook_id: str, data_file: tuple[str, bytes] | None = None) -> None:
    """
    Prepares the physical files needed for a docker build in a temporary directory.
    """
    path = Path(tmp_dir)
    
    # Generate main.py
    user_code = extract_code_from_notebook(notebook)
    main_py_content = generate_wrapper_code(user_code)
    (path / "main.py").write_text(main_py_content, encoding='utf-8')

    #  Write data.csv into tmp dir if exists in root folder
    
    # TESTING: local data.csv file (for testing purposes)
    data_csv_path = Path("data.csv")
    if data_csv_path.exists():
        shutil.copy(data_csv_path, path / "data.csv")

    # Write uploaded data file if provided (overwrites local data.csv if same name)
    if data_file:
        f_name, f_content = data_file
        (path / f_name).write_bytes(f_content)

    # DEBUG: Save generated main.py to local disk for inspection
    debug_dir = Path("debug_artifacts")
    debug_dir.mkdir(exist_ok=True)
    (debug_dir / f"{notebook_id}_main.py").write_text(main_py_content, encoding='utf-8')
    print(f"DEBUG: Saved generated wrapper to {debug_dir / f'{notebook_id}_main.py'}")
    
    # Generate requirements.txt
    reqs_txt = requirements if requirements else "numpy\ngrpcio\ngrpcio-tools\nprotobuf"
    (path / "requirements.txt").write_text(reqs_txt, encoding='utf-8')
    
    # copy Dockerfile
    dockerfile_template = RESOURCES_PATH / "Dockerfile.template"
    if dockerfile_template.exists():
        shutil.copy(dockerfile_template, path / "Dockerfile")
    else:
        df_content = "FROM python:3.10-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"main.py\"]"
        (path / "Dockerfile").write_text(df_content)

    # copy Protobufs
    proto_dir = path / "protos"
    proto_dir.mkdir(exist_ok=True)
    
    proto_files = ["common_pb2.py", "volpe_container_pb2.py", "volpe_container_pb2_grpc.py"]
    source_proto_dir = RESOURCES_PATH / "protos"
    
    for p_file in proto_files:
        src = source_proto_dir / p_file
        if not src.exists():
            src = VOLPE_PY_PATH / p_file
        
        if src.exists():
            shutil.copy(src, path / p_file)
        else:
            print(f"Warning: Proto file {p_file} not found.")

def create_build_context(notebook: Notebook, requirements: str | None, notebook_id: str, data_file: tuple[str, bytes] | None = None) -> io.BytesIO:
    """
    Performs 'docker build' and 'docker save' to return a full image tarball.
    """
    client = docker.from_env()
    image_tag = f"volpe-ea-{notebook_id.lower()}"
    
    # create a tmp dir for the build context
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Preparing build context in {tmp_dir}...")
        create_build_context_folder(tmp_dir, notebook, requirements, notebook_id, data_file)
        
        # docker Build
        print(f"Building Docker image {image_tag}...")
        try:
            image, build_logs = client.images.build(
                path=tmp_dir,
                tag=image_tag,
                rm=True
            )
            for line in build_logs:
                if 'stream' in line:
                    print(line['stream'].strip())
        except Exception as e:
            print(f"Docker build failed: {e}")
            raise

        # docker Save (as tar)
        print(f"Saving Docker image {image_tag} to tarball...")
        try:
            # image.save() returns a generator of bytes
            image_tar_stream = io.BytesIO()
            for chunk in image.save():
                image_tar_stream.write(chunk)
            
            image_tar_stream.seek(0)
            
            
            return image_tar_stream
        except Exception as e:
            print(f"Docker save failed: {e}")
            raise