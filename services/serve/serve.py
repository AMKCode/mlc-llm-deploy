import os
import subprocess
import json
import tvm
from mlc_llm.serve import EngineConfig, PopenServer

subprocess.run(['python', '-m', 'mlc_llm', 'compile',
                'Llama-3.1-8B-Instruct-q4f16_1-MLC/',
                '--opt', 'O3',
                '--overrides', 'disaggregation=1',
                '-o', 'Llama-3.1-8B-Instruct-q4f16_1-MLC/lib_disagg.so'])

# NOTE HOSTNAME only works in a pod
pod_name = os.getenv("HOSTNAME", default=None)
if pod_name is None:
    cuda_id = 0
else:
    cuda_id = pod_name.split("-")[-1]
    cuda_id = int(cuda_id)
print(f"cuda_id: {cuda_id}")

num_replicas = os.getenv("NUM_REPLICAS", default=None)
if num_replicas is None:
    num_replicas = 1
else:
    num_replicas = int(num_replicas)
print(f"num_replicas: {num_replicas}")

f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
uid = list(f_init_nvshmem_uid())

nvshmem_config = {
                "uid": uid,
                "npes": num_replicas,  # total number of workers in the nvshmem world
                "pe_start": cuda_id,  # start of PE for this endpoint's workers
                }

os.environ["MLC_NVSHMEM_INIT_CONFIG_JSON_STR"] = json.dumps(nvshmem_config)

subprocess.run(['python', '-m', 'mlc_llm', 'serve', 
                '/Llama-3.1-8B-Instruct-q4f16_1-MLC', 
                '--model-lib', '/Llama-3.1-8B-Instruct-q4f16_1-MLC/lib_disagg.so', 
                '--device', f'cuda:{cuda_id}', 
                '--enable-debug', 
                '--mode', 'server', 
                '--speculative-mode', 'disable', 
                '--prefix-cache-mode', 'disable', 
                '--overrides', 'gpu_memory_utilization=0.8;spec_draft_length=0', 
                '--host', '0.0.0.0', 
                '--port', '8000'])