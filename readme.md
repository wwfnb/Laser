# 安装 LLaMA-Factory
```
conda create --name llama_factory python==3.11
conda activate llama_factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
# 安装 Laser
```
pip install -e .
```

cache中保存了造数据，训练， 评估过程中 resize的image