import os
import subprocess
import json
import tvm
from mlc_llm.serve import EngineConfig, PopenServer

pod_name = os.getenv("POD_NAME", default=None)
if pod_name is None:
    print('echo "POD_NAME not found, defaulting CUDA_ID to 0"')
    cuda_id = 0
else:
    cuda_id = pod_name.split("-")[-1]
    print(f'echo "setting CUDA_ID to {cuda_id}"')
    cuda_id = int(cuda_id)

f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
uid = list(f_init_nvshmem_uid())

nvshmem_config = {
                "uid": uid,
                "npes": 1,  # total number of workers in the nvshmem world
                "pe_start": cuda_id,  # start of PE for this endpoint's workers
                }

os.environ["MLC_NVSHMEM_INIT_CONFIG_JSON_STR"] = json.dumps(nvshmem_config)

subprocess.run(['python', '-m', 'mlc_llm', 'serve', '/Llama-3.1-8B-Instruct-q4f16_1-MLC', '--model-lib', '/Llama-3.1-8B-Instruct-q4f16_1-MLC/lib_disagg.so', '--device', f'cuda:{cuda_id}', '--enable-debug', '--mode', 'server', '--speculative-mode', 'disable', '--prefix-cache-mode', 'disable', '--overrides', 'gpu_memory_utilization=0.8;spec_draft_length=0', '--enable-debug', '--host', '0.0.0.0', '--port', '8000'])