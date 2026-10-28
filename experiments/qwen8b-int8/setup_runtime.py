"""Prepare the pinned local runtime without changing the production environment."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess
import tarfile
import requests

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime'
TAG = 'b11944'
ARCHIVE = f'llama-{TAG}-bin-macos-arm64.tar.gz'
SHA256 = 'b7637fc703df818ce4779b9423c4caac9c29ab433770ba8d364913849af394f4'
HEADERS = {name: f'ggml/include/{name}' for name in [
    'ggml.h', 'ggml-cpu.h', 'ggml-backend.h', 'ggml-opt.h', 'gguf.h', 'ggml-alloc.h']}
HEADERS.update({'llama.h': 'include/llama.h', 'mtmd.h': 'tools/mtmd/mtmd.h',
                'mtmd-helper.h': 'tools/mtmd/mtmd-helper.h', 'json.hpp': 'vendor/nlohmann/json.hpp'})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    RUNTIME.mkdir(exist_ok=True)
    archive = RUNTIME / ARCHIVE
    if not archive.exists():
        url = f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{ARCHIVE}'
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with archive.open('wb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
    if digest(archive) != SHA256:
        raise ValueError('Runtime archive checksum mismatch')
    binary_dir = RUNTIME / f'llama-{TAG}'
    if not binary_dir.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(RUNTIME, filter='data')
    include = RUNTIME / 'include'
    include.mkdir(exist_ok=True)
    def fetch(pair):
        name, source = pair
        path = include / name
        if not path.exists():
            response = requests.get(f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{TAG}/{source}', timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
        return name, digest(path)
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        headers = dict(pool.map(fetch, HEADERS.items()))
    cmd = ['clang++', '-x', 'objective-c++', '-std=c++17', '-O2', '-I', str(include),
           str(HERE / 'aligned_encoder.cpp'), '-L', str(binary_dir),
           '-Wl,-rpath,' + str(binary_dir), '-lllama', '-lmtmd', '-lggml', '-lggml-base',
           '-framework', 'Metal', '-framework', 'Foundation', '-o', str(RUNTIME / 'aligned-encoder')]
    subprocess.run(cmd, check=True)
    (HERE / 'runtime-provenance.json').write_text(json.dumps({
        'tag': TAG, 'commit': '1b4554eff', 'archive_sha256': SHA256,
        'header_sha256': headers, 'compile_command': cmd,
        'adapter_source_sha256': digest(HERE / 'aligned_encoder.cpp'),
        'adapter_binary_sha256': digest(RUNTIME / 'aligned-encoder'),
    }, indent=2))
    print('Runtime ready:', RUNTIME / 'aligned-encoder')
