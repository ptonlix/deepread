# DeepRead

DeepRead 是一个智能文档处理系统，支持多种文档格式的转换、OCR 识别和内容分析。

## 功能特性

- **多格式文档支持**: PDF、DOCX、XLSX、HTML/HTM
- **智能 OCR 识别**: 集成 DeepSeek-OCR 和 vLLM 引擎
- **批量处理**: 支持大规模文档批量处理
- **多种输出格式**: Markdown、JSON、Rich Text（RTF）
- **Web 界面**: 基于 Streamlit 的用户友好界面
- **RESTful API**: FastAPI 支持程序化调用
- **命令行工具**: 便捷的 CLI 接口

## 系统要求

### 基础环境

- Python 3.12+
- 8GB+ RAM
- 10GB+ 磁盘空间

### GPU 加速（可选）

为了获得最佳 OCR 性能，推荐使用 GPU 加速：

- **NVIDIA GPU**: 支持 CUDA 11.8+
- **显存要求**: 8GB+ VRAM（推荐 16GB+）
- **驱动版本**: NVIDIA Driver 470+

支持的 GPU 型号：

- RTX 3080/3090/4080/4090
- Tesla V100/A100
- RTX A4000/A5000/A6000

## 安装指南

### 1. 基础安装

本项目使用 `uv` 进行依赖管理（推荐）：

```bash
# 克隆项目
git clone <repository-url>
cd deepread

# 安装基础依赖
uv sync --all-extras

# 安装pre-commit hooks（可选）
make githooks
```

如果没有 `uv`，可以使用传统方式：

```bash
pip install -e .
```

### 2. GPU 加速安装（推荐）

```bash
# 安装PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 确保vLLM已安装（已包含在依赖中）
# vLLM需要GPU支持
```

### 3. 开发环境安装

```bash
# 安装开发依赖
uv sync --all-extras

# 运行代码质量检查
make check

# 运行测试
make test
```

## 快速开始

### 命令行使用

DeepRead 提供了便捷的命令行接口：

```bash
# 提交文档处理任务
python -m deepread submit document.pdf --output-format markdown

# 批量处理多个文档
python -m deepread submit doc1.pdf doc2.docx doc3.xlsx --output-format markdown json

# 查看任务状态
python -m deepread status <job_id>

# 获取处理结果
python -m deepread fetch <job_id> --format markdown

# 查看任务清单
python -m deepread manifest <job_id>
```

**支持的命令**：

- `submit`: 提交文档处理任务，支持多个文档和多种输出格式
- `status`: 查看任务状态和所有子任务的进度
- `fetch`: 获取指定格式的处理结果
- `manifest`: 查看任务的完整清单

**支持的输出格式**：

- `markdown` (默认): Markdown 格式报告
- `json`: JSON 格式的结构化数据
- `rich_text`: RTF 格式的富文本报告

### Web 界面

```bash
# 启动Streamlit界面
streamlit run deepread/ui/app.py
```

访问 http://localhost:8501 使用 Web 界面。

Web 界面支持：

- 拖拽上传多个文档（PDF、DOCX、XLSX、HTML 等）
- 选择输出格式（Markdown、JSON、Rich Text）
- 查看处理状态和结果
- 下载处理结果和清单

### API 服务

```bash
# 启动API服务（推荐方式）
python -m deepread serve --host 0.0.0.0 --port 8000

# 或使用传统方式
uvicorn deepread.api.router:app --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
python -m deepread serve --reload
```

API 文档: http://localhost:8000/docs

**主要端点**：

- `POST /v1/jobs`: 提交批量文档处理任务
- `GET /v1/jobs/{job_id}`: 获取任务状态
- `GET /v1/reports/{submission_id}/content`: 获取指定格式的处理结果
- `GET /health`: 健康检查

## 配置说明

### 环境变量

```bash
# 工作区路径（存储处理结果和中间文件）
export DEEPREAD_STORE=/path/to/workspace

# OCR引擎模式
# 可选值: fallback, vllm_local, vllm_remote
export DEEPREAD_OCR_MODE=vllm_local

# vLLM本地配置
export DEEPREAD_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR
export DEEPREAD_OCR_TENSOR_PARALLEL_SIZE=1
export DEEPREAD_OCR_GPU_MEMORY_UTILIZATION=0.8
export DEEPREAD_OCR_MAX_MODEL_LEN=4096

# vLLM远程服务配置
export DEEPREAD_OCR_API_URL=http://localhost:8000/v1
export DEEPREAD_OCR_API_KEY=your-api-key
export DEEPREAD_OCR_TIMEOUT=30

# OCR处理参数
export DEEPREAD_OCR_MAX_TOKENS=2048
export DEEPREAD_OCR_TEMPERATURE=0.0
export DEEPREAD_OCR_CONFIDENCE_THRESHOLD=0.7
export DEEPREAD_OCR_MAX_RETRIES=3

# DeepSeek-OCR特定参数
export DEEPREAD_OCR_NGRAM_SIZE=30
export DEEPREAD_OCR_WINDOW_SIZE=90
```

### OCR 引擎选择

1. **vllm_local**: 本地 vLLM 引擎（需要 GPU 和 CUDA 支持）

   - 最高性能，适合单机部署
   - 需要充足的 GPU 显存

2. **vllm_remote**: 远程 vLLM 服务

   - 适合分布式部署
   - 需要可访问的 vLLM API 服务

3. **fallback**: 回退引擎（用于测试和开发）
   - 无需 GPU，适合快速测试
   - 不执行真实 OCR，仅用于验证流程

## 部署指南

### Docker 部署

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 安装Python和依赖
RUN apt-get update && apt-get install -y python3.12 python3-pip
COPY . /app
WORKDIR /app

# 安装uv和项目依赖
RUN pip install uv
RUN uv sync --all-extras

# 启动服务
CMD ["python", "-m", "deepread", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### GPU 服务器部署

```bash
# 1. 确认GPU可用
nvidia-smi

# 2. 安装CUDA工具包
# 参考: https://developer.nvidia.com/cuda-toolkit

# 3. 安装项目
uv sync --all-extras

# 4. 配置环境变量
export DEEPREAD_OCR_MODE=vllm_local
export CUDA_VISIBLE_DEVICES=0

# 5. 启动服务
python -m deepread serve --host 0.0.0.0 --port 8000
```

### 生产环境配置

```bash
# 使用gunicorn部署API
pip install gunicorn
gunicorn deepread.api.router:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# 使用nginx反向代理
# /etc/nginx/sites-available/deepread
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 性能优化

### GPU 内存优化

```bash
# 减少GPU内存使用
export DEEPREAD_OCR_GPU_MEMORY_UTILIZATION=0.6

# 限制模型序列长度
export DEEPREAD_OCR_MAX_MODEL_LEN=2048
```

### 批处理优化

处理大量文档时，系统会自动使用工作区管理来组织结果。每个任务都有独立的工作目录，便于跟踪和管理。

## 故障排除

### 常见问题

1. **CUDA 内存不足**

   ```bash
   # 减少GPU内存使用
   export DEEPREAD_OCR_GPU_MEMORY_UTILIZATION=0.6
   ```

2. **vLLM 安装失败**

   ```bash
   # 确认CUDA版本
   nvcc --version

   # 重新安装PyTorch
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **OCR 识别质量差**

   ```bash
   # 调整温度参数（增加随机性）
   export DEEPREAD_OCR_TEMPERATURE=0.1

   # 增加最大token数
   export DEEPREAD_OCR_MAX_TOKENS=4096

   # 降低置信度阈值
   export DEEPREAD_OCR_CONFIDENCE_THRESHOLD=0.6
   ```

4. **文档格式不支持**

   当前支持的格式：PDF、DOCX、XLSX、HTML/HTM。其他格式会返回错误信息。

### 日志调试

```bash
# 启用详细日志
export DEEPREAD_LOG_LEVEL=DEBUG

# 查看任务工作目录
ls -la $DEEPREAD_STORE/workspaces/
```

## 开发指南

### 项目结构

```
deepread/
├── cli/          # 命令行接口
│   └── commands.py
├── api/          # REST API
│   └── router.py
├── ui/           # Web界面（Streamlit）
│   └── app.py
├── ingest/       # 文档处理管道
│   ├── pipeline.py
│   ├── workspace.py
│   └── converters/  # 文档格式转换器
├── ocr/          # OCR引擎
│   └── deepseek.py
├── insights/     # 内容分析和报告生成
│   ├── models.py
│   ├── summarizer.py
│   └── templates.py
├── config.py     # 配置管理
└── __main__.py   # 主入口
```

### 运行测试

```bash
# 运行所有测试
make test

# 或直接使用pytest
uv run pytest

# 运行特定测试
uv run pytest tests/unit/test_config.py

# 运行集成测试
uv run pytest tests/integration/

# 查看测试覆盖率
uv run pytest --cov deepread --cov-report=html
```

### 代码质量

```bash
# 运行完整的代码质量检查（lint + typecheck + deptry）
make check

# 仅运行类型检查
make typecheck

# 代码格式化（Ruff）
uv run ruff format .

# 代码检查（Ruff）
uv run ruff check .
```

### 构建和发布

```bash
# 构建wheel文件
make build

# 清理构建产物
make clean-build

# 发布到PyPI（需要配置UV_PUBLISH_TOKEN）
make publish
```

## 工作流程

DeepRead 的处理流程如下：

1. **文档转换**: 将输入文档（PDF、DOCX 等）转换为页面图像
2. **OCR 识别**: 使用 DeepSeek-OCR 对每页图像进行文字识别
3. **内容分析**: 聚合所有页面的 OCR 结果，生成结构化报告
4. **格式输出**: 根据请求的格式（Markdown、JSON、RTF）生成输出文件

每个处理任务都会在指定的工作区创建独立目录，包含：

- 原始文档副本
- 页面图像文件
- OCR 结果
- 最终输出文件（多种格式）
- 处理清单（manifest.md）

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request！请参考：

- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- 提交 PR 前请确保：
  - 运行 `make check` 通过所有代码质量检查
  - 运行 `make test` 通过所有测试
  - 遵循项目的代码风格和提交规范

## 支持

如有问题，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/your-org/deepread/issues)
- 文档: 查看项目 `docs/` 目录下的文档

## 技术栈

- **Python 3.12+**: 核心语言
- **FastAPI**: RESTful API 框架
- **Streamlit**: Web 界面
- **vLLM**: 高性能 LLM 推理引擎
- **DeepSeek-OCR**: OCR 模型
- **Pillow**: 图像处理
- **pypdf**: PDF 处理
- **python-docx**: DOCX 文档处理
- **openpyxl**: Excel 文档处理
- **BeautifulSoup4**: HTML 解析
- **uv**: 现代化 Python 包管理
