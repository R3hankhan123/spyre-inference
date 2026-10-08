# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builds the host CPU kernels (csrc/) with CMake, as vLLM's setup.py does.

Metadata lives in pyproject.toml. Supported hosts: x86_64, ppc64le, s390x.
"""

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class cmake_build_ext(build_ext):
    def build_extensions(self) -> None:
        # The kernels are optional: without them the samplers fall back to
        # PyTorch ops, so a failed build must not fail the install.
        try:
            if not self._restore_from_cache():
                self._build_with_cmake()
                self._store_in_cache()
        except (OSError, ImportError, subprocess.CalledProcessError) as e:
            cmake_failed = isinstance(e, subprocess.CalledProcessError)
            why = "see the CMake output above" if cmake_failed else e
            print(
                f"WARNING: spyre-inference sampling kernels failed to build ({why}). "
                "Installing without them: host sampling falls back to PyTorch ops "
                "and gets no kernel speedup.",
                file=sys.stderr,
            )
            # setuptools copies and lists outputs from this list; nothing was built.
            self.extensions = []

    # SPYRE_EXT_CACHE_DIR keys built extensions on a hash of their inputs, so CI
    # can reuse them across fresh checkouts (uv's own cache keys on paths/mtimes).
    def _cache_entry(self) -> Path | None:
        cache_dir = os.environ.get("SPYRE_EXT_CACHE_DIR")
        if not cache_dir:
            return None
        h = hashlib.sha256()
        inputs = [ROOT / "CMakeLists.txt", ROOT / "setup.py"]
        inputs += sorted(p for d in ("cmake", "csrc") for p in (ROOT / d).rglob("*") if p.is_file())
        for path in inputs:
            h.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
        for part in (
            importlib.metadata.version("torch"),
            platform.machine(),
            sysconfig.get_config_var("EXT_SUFFIX"),
            os.environ.get("CMAKE_BUILD_TYPE", "Debug" if self.debug else "RelWithDebInfo"),
            os.environ.get("CMAKE_ARGS", ""),
        ):
            h.update(str(part).encode() + b"\0")
        return Path(cache_dir) / h.hexdigest()[:16]

    def _outputs(self) -> list[Path]:
        return [Path(self.get_ext_fullpath(ext.name)) for ext in self.extensions]

    def _restore_from_cache(self) -> bool:
        entry = self._cache_entry()
        if entry is None or not all((entry / out.name).is_file() for out in self._outputs()):
            return False
        for out in self._outputs():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry / out.name, out)
        print(f"Reused cached spyre-inference kernels from {entry}")
        return True

    def _store_in_cache(self) -> None:
        entry = self._cache_entry()
        if entry is None:
            return
        entry.mkdir(parents=True, exist_ok=True)
        for out in self._outputs():
            tmp = entry / f".{out.name}.tmp"
            shutil.copy2(out, tmp)
            os.replace(tmp, entry / out.name)

    def _build_with_cmake(self) -> None:
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = os.environ.get("CMAKE_BUILD_TYPE", "Debug" if self.debug else "RelWithDebInfo")
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DSPYRE_PYTHON_EXECUTABLE={sys.executable}",
            f"-DSPYRE_PYTHON_PATH={':'.join(sys.path)}",
        ]
        if extra := os.environ.get("CMAKE_ARGS"):
            cmake_args += extra.split()
        subprocess.check_call(["cmake", str(ROOT), *cmake_args], cwd=build_temp)

        targets = [ext.name.removeprefix("spyre_inference.") for ext in self.extensions]
        # Two sources per target, so more jobs only oversubscribe a shared runner.
        num_jobs = os.environ.get("MAX_JOBS") or str(2 * len(targets))
        subprocess.check_call(
            ["cmake", "--build", ".", f"-j={num_jobs}"] + [f"--target={t}" for t in targets],
            cwd=build_temp,
        )

        for ext, target in zip(self.extensions, targets):
            # CMake appends DESTINATION (the package dir) to the prefix.
            prefix = Path(self.get_ext_fullpath(ext.name)).parent.parent.resolve()
            subprocess.check_call(
                ["cmake", "--install", ".", "--prefix", str(prefix), "--component", target],
                cwd=build_temp,
            )


ext_modules = [CMakeExtension("spyre_inference._C")]
if platform.machine() == "x86_64":
    ext_modules.append(CMakeExtension("spyre_inference._C_AVX2"))

setup(ext_modules=ext_modules, cmdclass={"build_ext": cmake_build_ext})
