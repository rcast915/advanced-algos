# Toolchain image for the DARP project: CUDA 13 (nvcc, sm_90 for the H100), g++ with OpenMP, Python + numpy/scipy/pandas/matplotlib.
# Same CUDA base as ds_experiment. The repo is NOT copied in; it is bind-mounted at /work by submit_punakha.sh.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      make g++ python3 python3-numpy python3-scipy python3-pandas python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
CMD ["/bin/bash"]
