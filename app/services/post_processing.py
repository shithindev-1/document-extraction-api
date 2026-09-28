"""Parsing Gemini's extraction text and building /ocr response/error payloads."""

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app.schemas.ocr import (
    DATA_FIELDS,
    DOCUMENT_TYPE_ALIASES,
    FRONT_BACK_TYPES,
    MISSING_INFO_FIELDS,
    VERIFIABLE_TYPES,
    ErrorInfo,
    OcrResponse,
)


_FIELD_LOOKUP = {field.lower(): field for field in DATA_FIELDS}
_SIDE_KEYS = {"front_side_visible": "front", "back_side_visible": "back"}
_TYPES_KEY = "document_types_present"
_TYPE_ALIASES = DOCUMENT_TYPE_ALIASES
_TRUE_VALUES = {"true", "yes", "y", "visible", "present", "1"}
_NULL_VALUES = {
    "",
    "-",
    "--",
    "null",
    "none",
    "nil",
    "n/a",
    "na",
    "unknown",
    "not visible",
    "not present",
    "not available",
    "not specified",
    "not readable",
    "unreadable",
}


def _clean_key(raw_key: str) -> str:
    key = raw_key.strip().lstrip("-*•#> \t").strip()
    return key.replace("*", "").replace("`", "").strip().lower().replace(" ", "_")


def _clean_value(raw_value: str) -> str | None:
    value = raw_value.strip().replace("**", "").strip("`").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return None if value.lower() in _NULL_VALUES else (value or None)


def _parse_types(raw_value: str) -> list[str]:
    """Read the Document_Types_Present line into normalised type names.

    Tolerant in the same way the field parser is: anything unrecognised is dropped rather than
    failing the line, and a few obvious spellings of the same document are folded together, so a
    stray "Emirates ID" does not read as a different type from "national_id".
    """
    types: list[str] = []
    for token in (raw_value or "").replace(";", ",").split(","):
        name = token.strip().strip(".").lower().replace(" ", "_").replace("-", "_")
        name = _TYPE_ALIASES.get(name, name)
        if name in VERIFIABLE_TYPES or name == "other":
            if name not in types:
                types.append(name)
    return types


def parse_extraction_text(text: str) -> dict[str, Any]:
    """Turn Gemini's plain `Field_Name: value` output into the structure the response is built
    from. Deliberately line-tolerant: any line that isn't a recognised field is skipped, so a
    stray markdown fence, heading, or sentence of preamble degrades to a skipped line instead of
    failing the whole extraction the way a single bad character breaks json.loads."""
    data: dict[str, Any] = dict.fromkeys(DATA_FIELDS)
    # Gemini reports what it can see on each side; completeness is concluded here, not by the
    # model. Each side defaults to visible so a response that omits the lines entirely — which
    # is the norm for document types that have no back — is not treated as incomplete.
    sides = {"front": True, "back": True}
    # Empty means the model did not report - which is not the same as "no types found", so the
    # caller treats it as unknown and does not block on it.
    types_present: list[str] = []
    matched = 0

    for line in text.splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = _clean_key(raw_key)
        if key in _FIELD_LOOKUP:
            data[_FIELD_LOOKUP[key]] = _clean_value(raw_value)
            matched += 1
        elif key in _SIDE_KEYS:
            value = _clean_value(raw_value)
            sides[_SIDE_KEYS[key]] = value is not None and value.lower() in _TRUE_VALUES
            matched += 1
        elif key == _TYPES_KEY:
            types_present = _parse_types(raw_value)
            matched += 1

    if matched == 0:
        # Nothing recognisable at all means a malformed model response, not a blank document —
        # a genuinely unreadable document still yields "Field_Name: null" lines.
        raise ValueError("Extraction response contained no recognisable field lines")

    return {
        "data": data,
        "documentComplete": sides["front"] and sides["back"],
        "documentTypesPresent": types_present,
        # Internal only - never reaches the API response. Lets the caller tell "shows the front"
        # apart from "shows the back" apart from "shows neither", so an incomplete document can be
        # matched against another incomplete upload of the same type that supplies its missing side,
        # instead of every incomplete document being rejected outright.
        "frontVisible": sides["front"],
        "backVisible": sides["back"],
    }


def _type_mismatch_error(document_name: str, filename: str, document_type: str, found: list[str]) -> dict[str, str]:
    label = document_type.replace("_", " ").title()
    seen = ", ".join(name.replace("_", " ").title() for name in found if name != "other")
    detail = f"the file contains {seen}" if seen else "no document of that type was found in the file"
    return {
        "DocumentName": document_name,
        "DocumentFileName": filename,
        "DocumentError": f"Uploaded document '{filename}' does not match the selected document type {label}: {detail}.",
        "DocumentErrorToShow": f"'{filename}' is not a {label}. Select the correct document type or upload the matching document.",
    }


def document_type_matches(document_type: str, types_present: Any) -> bool:
    """Whether the selected type is actually present in the uploaded file.

    Only the three identity documents can be checked - anything else is accepted as declared. An
    empty or malformed list means the model did not report, which is treated as unknown rather
    than as a mismatch: a missing observation must never reject a document the user did upload.
    A file showing several types passes as soon as the selected one is among them, and extraction
    then reads that document, exactly as the system instruction already directs.
    """
    if document_type not in VERIFIABLE_TYPES:
        return True
    if not isinstance(types_present, list) or not types_present:
        return True
    return document_type in types_present


def _incomplete_error(document_name: str, filename: str, document_type: str) -> dict[str, str]:
    label = document_type.replace("_", " ").title()
    return {
        "DocumentName": document_name,
        "DocumentFileName": filename,
        "DocumentError": f"{label} must contain both FRONT and BACK in the same file.",
        "DocumentErrorToShow": f"Incomplete {label}: FRONT and BACK are required in the same file.",
    }


def normalize_response(
    raw_response: Mapping[str, Any],
    *,
    document_name: str,
    document_filename: str,
    document_type: str,
    default_reference_id: str | None = None,
) -> dict[str, Any]:
    raw_data = raw_response.get("data", {})
    data = {
        field: raw_data.get(field) if isinstance(raw_data, Mapping) else None
        for field in DATA_FIELDS
    }
    candidate = {
        "success": True,
        "userName": raw_response.get("userName"),
        "tenantType": raw_response.get("tenantType"),
        "subscriptionId": raw_response.get("subscriptionId"),
        "ocrReferenceId": raw_response.get("ocrReferenceId") or default_reference_id,
        "data": data,
        "missingInfo": [
            missing_name
            for field, missing_name in zip(DATA_FIELDS, MISSING_INFO_FIELDS)
            if data[field] is None
        ],
        "errorInfo": raw_response.get("errorInfo", []),
    }

    try:
        validated = OcrResponse.model_validate(candidate)
    except ValidationError:
        validated = OcrResponse(data=data)

    errors = [error.model_dump() for error in validated.errorInfo]
    types_present = raw_response.get("documentTypesPresent")
    if not document_type_matches(document_type, types_present):
        errors = [_type_mismatch_error(document_name, document_filename, document_type, types_present or [])]
    elif raw_response.get("documentComplete") is False and document_type in FRONT_BACK_TYPES:
        # Guarded by type as well as by the prompt: a stray documentComplete=false for a document
        # that has no front/back requirement must not turn into an incomplete-document error.
        errors = [_incomplete_error(document_name, document_filename, document_type)]

    if errors:
        # Any error found here - the wrong document was uploaded, or the right one but missing its
        # required back side - means nothing extracted is trustworthy enough to hand back. Reporting
        # a problem and returning the data alongside it would let a caller silently use fields read
        # off an incomplete or wrong document, so the response never carries both: the data is
        # dropped entirely, which also makes success false.
        validated = OcrResponse(ocrReferenceId=candidate["ocrReferenceId"])
    validated.errorInfo = [ErrorInfo.model_validate(error) for error in errors]
    return validated.model_dump()


def incomplete_response(
    *, document_name: str, document_filename: str, document_type: str, reference_id: str | None = None,
) -> dict[str, Any]:
    return OcrResponse(
        ocrReferenceId=reference_id,
        errorInfo=[ErrorInfo.model_validate(_incomplete_error(document_name, document_filename, document_type))],
    ).model_dump()


def _ambiguous_pair_error(document_name: str, filename: str, document_type: str, count: int) -> dict[str, str]:
    label = document_type.replace("_", " ").title()
    return {
        "DocumentName": document_name,
        "DocumentFileName": filename,
        "DocumentError": f"{count} incomplete {label} uploads were found and could not be automatically "
                          f"matched into front/back pairs.",
        "DocumentErrorToShow": f"We received {count} incomplete {label} uploads and couldn't tell which "
                                f"front matches which back. Please upload each {label}'s front and back "
                                f"together in one file, or make sure only one incomplete {label} is "
                                f"uploaded at a time.",
    }


def ambiguous_pair_response(
    *, document_name: str, document_filename: str, document_type: str, count: int, reference_id: str | None = None,
) -> dict[str, Any]:
    return OcrResponse(
        ocrReferenceId=reference_id,
        errorInfo=[ErrorInfo.model_validate(
            _ambiguous_pair_error(document_name, document_filename, document_type, count)
        )],
    ).model_dump()


def _pair_mismatch_error(document_name: str, filename: str, document_type: str) -> dict[str, str]:
    label = document_type.replace("_", " ").title()
    return {
        "DocumentName": document_name,
        "DocumentFileName": filename,
        "DocumentError": f"Two incomplete {label} uploads were provided as a front/back pair, but they do "
                          f"not appear to be the same document.",
        "DocumentErrorToShow": f"The two {label} uploads you provided don't appear to be the front and back "
                                f"of the same document. Please check and re-upload matching files.",
    }


def pair_mismatch_response(
    *, document_name: str, document_filename: str, document_type: str, reference_id: str | None = None,
) -> dict[str, Any]:
    return OcrResponse(
        ocrReferenceId=reference_id,
        errorInfo=[ErrorInfo.model_validate(
            _pair_mismatch_error(document_name, document_filename, document_type)
        )],
    ).model_dump()


def error_response(
    *,
    document_name: str,
    document_filename: str,
    message: str,
    message_to_show: str | None = None,
) -> dict[str, Any]:
    """Same envelope every other endpoint response uses, so a rejected request never
    forces a caller to branch on a different, bare `{"detail": ...}` shape."""
    return OcrResponse(
        errorInfo=[
            ErrorInfo(
                DocumentName=document_name,
                DocumentFileName=document_filename,
                DocumentError=message,
                DocumentErrorToShow=message_to_show or message,
            )
        ]
    ).model_dump()