# ocr Specification

## Purpose
TBD - created by archiving change add-deepseek-ocr-integration. Update Purpose after archive.
## Requirements
### Requirement: DeepSeek OCR Integration
The ingestion pipeline MUST use DeepSeek-OCR via a vLLM inference backend to extract text and confidences from rendered page images.

#### Scenario: Successful recognition
- **GIVEN** a rendered page image produced during document ingestion
- **AND** a reachable vLLM-powered DeepSeek-OCR inference endpoint
- **WHEN** DeepSeek-OCR finishes inference without errors
- **THEN** the pipeline records the extracted text and confidence on the corresponding `PageInsight`
- **AND** warnings remain empty unless retries were required

### Requirement: OCR Failure Fallback
The system MUST provide deterministic fallback text and remediation guidance when DeepSeek-OCR cannot return a confident result.

#### Scenario: Engine failure
- **GIVEN** DeepSeek-OCR raises an exception while processing a page image
- **WHEN** the pipeline handles the failure
- **THEN** it logs a warning, stores fallback text derived from the original document bytes, and marks the submission with remediation details

