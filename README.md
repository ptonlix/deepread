# DeepRead

DeepRead 是一个智能文档处理系统，支持多种文档格式的转换、OCR识别和内容分析。

## 功能特性

- **多格式文档支持**: PDF、DOCX、XLSX、HTML等格式
- **智能OCR识别**: 集成DeepSeek OCR和vLLM引擎
- **批量处理**: 支持大规模文档批量处理
- **多种输出格式**: JSON、Markdown、富文本等
- **Web界面**: 基于Streamlit的用户友好界面
- **API接口**: RESTful API支持程序化调用

## 系统要求

### 基础环境

- Python 3.12+
- 8GB+ RAM
- 10GB+ 磁盘空间

### GPU加速（可选）

为了获得最佳OCR性能，推荐使用GPU加速：

- **NVIDIA GPU**: 支持CUDA 11.8+
- **显存要求**: 8GB+ VRAM（推荐16GB+）
- **驱动版本**: NVIDIA Driver 470+

支持的GPU型号：

- RTX 3080/3090/4080/4090
- Tesla V100/A100
- RTX A4000/A5000/A6000

## 安装指南

### 1. 基础安装

```bash
# 克隆项目
git clone <repository-url>
cd deepread

# 安装基础依赖
pip install -e .
```

### 2. GPU加速安装（推荐）

```bash
# 安装PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装vLLM（需要GPU）
pip install -e ".[vllm]"
```

### 3. 开发环境安装

```bash
# 安装开发依赖
pip install -e ".[dev-full]"

# 安装pre-commit hooks
pre-commit install
```

## 快速开始

### 命令行使用

```bash
# 处理单个文档
python -m deepread process document.pdf

# 批量处理
python -m deepread batch /path/to/documents/

# 指定输出格式
python -m deepread process document.pdf --format json

# 使用vLLM引擎
python -m deepread process document.pdf --ocr-mode vllm_local
```

### Web界面

```bash
# 启动Streamlit界面
streamlit run deepread/ui/app.py
```

访问 http://localhost:8501 使用Web界面。

### API服务

```bash
# 启动API服务
python -m deepread api

# 或使用uvicorn
uvicorn deepread.api.router:app --host 0.0.0.0 --port 8000
```

API文档: http://localhost:8000/docs

## 配置说明

### 环境变量

```bash
# 工作区路径
export DEEPREAD_STORE=/path/to/workspace

# OCR配置
export DEEPREAD_OCR_MODE=vllm_local  # 或 vllm_remote, echo
export DEEPREAD_OCR_MODEL=deepseek-ai/deepseek-vl-7b-chat
export DEEPREAD_OCR_MAX_TOKENS=2048
export DEEPREAD_OCR_TEMPERATURE=0.0

# vLLM远程服务配置
export DEEPREAD_VLLM_BASE_URL=http://localhost:8000/v1
export DEEPREAD_VLLM_TIMEOUT=300
```

### OCR引擎选择

1. **vllm_local**: 本地vLLM引擎（需要GPU）
2. **vllm_remote**: 远程vLLM服务
3. **echo**: 回退引擎（用于测试）

## 部署指南

### Docker部署

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 安装Python和依赖
RUN apt-get update && apt-get install -y python3.12 python3-pip
COPY . /app
WORKDIR /app

# 安装依赖
RUN pip install -e ".[vllm]"

# 启动服务
CMD ["python", "-m", "deepread", "api"]
```

### GPU服务器部署

```bash
# 1. 确认GPU可用
nvidia-smi

# 2. 安装CUDA工具包
# 参考: https://developer.nvidia.com/cuda-toolkit

# 3. 安装项目
pip install -e ".[vllm]"

# 4. 启动服务
export CUDA_VISIBLE_DEVICES=0
python -m deepread api --host 0.0.0.0 --port 8000
```

### 生产环境配置

```bash
# 使用gunicorn部署API
pip install gunicorn
gunicorn deepread.api.router:app -w 4 -k uvicorn.workers.UvicornWorker

# 使用nginx反向代理
# /etc/nginx/sites-available/deepread
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 性能优化

### GPU内存优化

```python
# 配置vLLM引擎
config = {
    "model_name": "deepseek-ai/deepseek-vl-7b-chat",
    "gpu_memory_utilization": 0.8,  # 使用80%显存
    "max_model_len": 4096,          # 限制序列长度
    "tensor_parallel_size": 1,      # 单GPU
}
```

### 批处理优化

```bash
# 增加并发处理数
export DEEPREAD_MAX_WORKERS=4

# 调整批处理大小
export DEEPREAD_BATCH_SIZE=8
```

## 故障排除

### 常见问题

1. **CUDA内存不足**

   ```bash
   # 减少GPU内存使用
   export DEEPREAD_OCR_GPU_MEMORY_UTILIZATION=0.6
   ```

2. **vLLM安装失败**

   ```bash
   # 确认CUDA版本
   nvcc --version

   # 重新安装PyTorch
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **OCR识别质量差**

   ```bash
   # 调整温度参数
   export DEEPREAD_OCR_TEMPERATURE=0.1

   # 增加最大token数
   export DEEPREAD_OCR_MAX_TOKENS=4096
   ```

### 日志调试

```bash
# 启用详细日志
export DEEPREAD_LOG_LEVEL=DEBUG

# 查看OCR处理日志
python -m deepread process document.pdf --verbose
```

## 开发指南

### 项目结构

```
deepread/
├── cli/          # 命令行接口
├── api/          # REST API
├── ui/           # Web界面
├── ingest/       # 文档处理管道
├── ocr/          # OCR引擎
├── insights/     # 内容分析
└── config.py     # 配置管理
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_config.py

# 运行集成测试
pytest tests/integration/
```

### 代码质量

```bash
# 代码格式化
ruff format

# 代码检查
ruff check

# 类型检查
mypy deepread/
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献指南

欢迎提交Issue和Pull Request！请参考 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 支持

如有问题，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/your-org/deepread/issues)
- 邮件: support@deepread.ai
- 文档: [在线文档](https://docs.deepread.ai)
