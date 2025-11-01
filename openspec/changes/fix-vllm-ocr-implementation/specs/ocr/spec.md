## MODIFIED Requirements

### Requirement: DeepSeek OCR Integration

The ingestion pipeline MUST use DeepSeek-OCR via a vLLM inference backend to extract text and confidences from rendered page images.

The vLLM integration SHALL follow the official DeepSeek-OCR-vllm implementation patterns:

- Register the `DeepseekOCRForCausalLM` model class with vLLM's ModelRegistry before engine initialization
- Use `DeepseekOCRProcessor` for proper image preprocessing and tokenization
- Use `NoRepeatNGramLogitsProcessor` to suppress repetitive text generation
- Configure vLLM with `hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]}` and other required parameters
- Format image inputs as `{"prompt": str, "multi_modal_data": {"image": processed_image_data}}`

#### Scenario: Successful recognition

- **GIVEN** a rendered page image produced during document ingestion
- **AND** a properly configured vLLM-powered DeepSeek-OCR inference engine with the model registered
- **WHEN** DeepSeek-OCR finishes inference without errors
- **THEN** the pipeline records the extracted text and confidence on the corresponding `PageInsight`
- **AND** warnings remain empty unless retries were required
- **AND** the image was preprocessed using `DeepseekOCRProcessor` before inference

#### Scenario: Engine initialization

- **GIVEN** a vLLM local engine is requested
- **WHEN** the engine is initialized
- **THEN** `DeepseekOCRForCausalLM` is registered with vLLM's ModelRegistry
- **AND** the LLM instance is created with `hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]}`
- **AND** the engine is ready to process images
