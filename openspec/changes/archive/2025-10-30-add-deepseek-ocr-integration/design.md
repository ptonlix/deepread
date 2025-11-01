## Overview

将DeepSeek-OCR集成到现有的`ProcessingPipeline`，提供真实的文档识别能力，并保持原有的接口稳定性。目标是在保证吞吐量和稳定性的前提下引入基于vLLM的GPU推理，允许在不同部署场景下切换推理模式。

## Architecture

1. **OCR引擎封装**

   - 在`deepread.ocr.deepseek`中实现`DeepSeekInferenceEngine`，负责构造vLLM客户端、图像预处理、推理和置信度计算。
   - 本地模式下通过vLLM Python API加载DeepSeek-OCR权重到GPU，并持久化engine实例；远程模式下通过HTTP/GRPC调用vLLM Inference服务。
   - 提供同步接口以便在`asyncio.to_thread`中运行；如未来需要异步化，可通过请求队列或连接池封装。

2. **管道集成**

   - `_run_render_and_ocr`复用单个`DeepSeekOcr`实例，在每页图像执行推理并写入warnings；该实例内部管理vLLM Session或HTTP连接。
   - 支持两种模式：
     - _本地模式_: 使用vLLM Python API（`LLMEngine` / `AsyncLLMEngine`）加载DeepSeek-OCR模型，直接在进程内推理。
     - _远程模式_: 调用基于vLLM Inference部署的官方server（FastAPI+Uvicorn），参数包括服务URL、超时、重试次数。
   - 在环境变量 / 配置文件中暴露模式选择、vLLM配置（模型路径、并发限制、tensor parallel）等参数。

3. **失败处理与降级**
   - 保持现有fallback：推理失败时返回默认文本，并在`warnings`与`remediation`中记录具体原因。
   - 对置信度低于阈值的结果触发重试，可调整prompt或使用增强模式。

## Dependencies

- PyTorch >= 2.x、CUDA/cuDNN、transformers、accelerate、opencv-python-headless、safetensors、vllm >= 0.5等。
- 需要下载DeepSeek-OCR的`deepseek-text-ocr`模型参数（detector、recognizer、text 模型），并按vLLM格式组织（例如safetensors+tokenizer）。
- 若使用远程服务，需要FastAPI + Uvicorn部署vLLM Inference server，或复用DeepSeek-OCR仓库中的部署脚本。

## Open Questions

- 是否需要将OCR推理与主进程隔离（例如独立vLLM服务或进程池）以提升稳定性？
- 线上环境是否具备GPU资源，若无是否提供CPU降级模式（性能极低，需明确不推荐）？
- 权重文件的分发策略（容器镜像预置 vs. 启动时下载）。
- 推理日志与监控的集成方式（例如Prometheus指标、请求追踪）。**_ End Patch_**
