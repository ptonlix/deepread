"""Streamlit experience for the document OCR insight pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from deepread.cli.commands import SUPPORTED_FORMATS
from deepread.ingest.pipeline import ProcessingPipeline

st.set_page_config(page_title="deepread Insights", layout="wide")
st.title("Document OCR Insight Pipeline")


@st.cache_resource
def _pipeline() -> ProcessingPipeline:
    workspace_root = (
        Path(os.environ.get("DEEPREAD_STORE", ".deepread-store")) / "streamlit"
    )
    return ProcessingPipeline(workspace_root=workspace_root)


uploaded_files = st.file_uploader(
    "Upload documents",
    type=list(SUPPORTED_FORMATS) + ["pdf", "docx", "xlsx", "html", "htm"],
    accept_multiple_files=True,
)

selected_formats = st.multiselect(
    "Select output formats",
    options=sorted(SUPPORTED_FORMATS),
    default=["markdown"],
)

if st.button("Process Batch"):
    if not uploaded_files:
        st.warning("Please upload at least one document to process.")
    else:
        documents = [(file.read(), file.name) for file in uploaded_files]
        pipeline = _pipeline()
        result = pipeline.process_batch(
            documents=documents, requested_formats=set(selected_formats)
        )

        st.success(f"Job {result.job_id} completed with status: {result.status}")

        table = pd.DataFrame(
            [
                {
                    "Submission": submission.original_filename,
                    "Status": submission.status,
                    "Outputs": ", ".join(submission.outputs.keys()) or "—",
                    "Remediation": submission.remediation or "—",
                }
                for submission in result.submissions
            ]
        )
        st.dataframe(table, use_container_width=True)

        manifest_content = Path(result.manifest_path).read_text(encoding="utf-8")
        st.download_button(
            label="Download Manifest",
            data=manifest_content,
            file_name="manifest.md",
            mime="text/markdown",
        )
