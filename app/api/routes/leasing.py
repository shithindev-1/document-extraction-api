"""POST /ocr/leasing and /ocr/leasing/upload - the same pipeline as /ocr, leasing response shape.

  - `POST /ocr/leasing`         JSON body, documents named by URL - the API downloads each one.
  - `POST /ocr/leasing/upload`  multipart form-data, documents attached directly as files.

Both return `LeasingOcrResponse`. See app.services.leasing for how a batch is judged.
"""

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.leasing import LeasingOcrRequest, LeasingOcrResponse
from app.services.leasing import process_batch, process_sources

router = APIRouter(prefix="/ocr/leasing", tags=["leasing"])


@router.post("", response_model=LeasingOcrResponse)
async def extract_ocr_leasing(request: LeasingOcrRequest) -> dict[str, Any]:
    """Run the OCR pipeline over documents given as downloadable URLs."""
    return await process_sources(request)


@router.post("/upload", response_model=LeasingOcrResponse)
async def extract_ocr_leasing_upload(
    tenant_type: Annotated[str, Form(...)],
    document_refnumber: Annotated[str, Form(...)],
    file: Annotated[list[UploadFile], File(...)],
    document_id: Annotated[list[str], Form(...)],
    document_type: Annotated[list[str], Form(...)],
    document_name: Annotated[list[str], Form(...)],
) -> dict[str, Any]:
    """Run the OCR pipeline over documents attached as multipart files.

    Send one `file`, `document_id`, `document_type` and `document_name` field per document, all
    using those same repeated key names - they are paired up by position, first `file` with first
    `document_id` and so on. `tenant_type` and `document_refnumber` are sent once per request.
    """
    if not len(file) == len(document_id) == len(document_type) == len(document_name):
        raise HTTPException(
            status_code=400,
            detail="Each uploaded file must be sent with its own document_id, document_type, and document_name",
        )

    items: list[dict[str, Any]] = []
    for upload, doc_id, doc_type, doc_name in zip(file, document_id, document_type, document_name):
        content = await upload.read()
        items.append({
            "source": upload.filename or doc_name,
            "content": content,
            "filename": upload.filename or doc_name,
            "content_type": upload.content_type or "",
            "document_id": doc_id,
            "document_type": doc_type,
            "document_name": doc_name,
        })

    return await process_batch(tenant_type=tenant_type, document_refnumber=document_refnumber, items=items)
