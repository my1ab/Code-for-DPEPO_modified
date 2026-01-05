conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop

cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd /data/home/zhangjs/disk/project/verl-agent
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2

# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.