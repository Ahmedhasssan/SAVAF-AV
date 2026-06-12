FROM rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0

WORKDIR /workspace

RUN pip install --no-cache-dir \
    amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/ && \
    pip install --no-cache-dir \
    plyfile \
    tqdm \
    lpips \
    tensorboard \
    scipy \
    opencv-python-headless \
    matplotlib \
    wandb \
    einops \
    librosa \
    soundfile \
    scikit-video \
    jaxtyping

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["bash"]
