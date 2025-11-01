## 1. 准备

- [ ] 1.1 调研 DeepSeek-OCR 官方推理接口https://github.com/deepseek-ai/DeepSeek-OCR/，重点评估vLLM Inference 部署（本地/远程）所需步骤与模型文件
- [ ] 1.2 在开发环境验证依赖安装（PyTorch、transformers、vllm、safetensors 等）并记录 GPU 与 CUDA 要求

## 2. 实现

- [ ] 2.1 为`DeepSeekOcr`新增基于 vLLM 的`InferenceEngine`实现，负责模型加载、会话复用与推理请求
- [ ] 2.2 更新`ProcessingPipeline`以复用新的 OCR 实例，并保留失败时的回退逻辑
- [ ] 2.3 扩展配置/环境变量，支持切换推理模式（本地、远程服务、禁用）

## 3. 测试与验证

- [ ] 3.1 编写单元测试覆盖引擎失败与重试逻辑（可使用假引擎）
- [ ] 3.2 编写端到端测试脚本/文档，使用样例文档验证生成的 Markdown/JSON/RTF 内容
- [ ] 3.3 在 GPU 环境运行`make test`并记录性能与资源消耗

## 4. 文档与发布

- [ ] 4.1 更新 README/部署指引，说明新增依赖、环境准备和故障排查
- [ ] 4.2 若需要，调整 Streamlit/CLI 提示文案，提醒用户 GPU 与模型要求
- [ ] 4.3 最终校对提案，提交变更并请求评审
