"""Application settings, loaded from environment variables and `.env`."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str
    gemini_max_tokens: int = 2048
    # Per Gemini call, not per request. A call that runs longer fails like any other provider
    # error, so a hung call can no longer hold a request open indefinitely.
    gemini_timeout_seconds: int = 105
    # Gemini scales oversized images down and bills them in 768px tiles anyway, so a larger
    # original adds no detail - only upload bandwidth and inline-payload size.
    gemini_max_image_edge: int = 1568
    require_https: bool = True
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    max_file_size_mb: int = 15
    # The whole box-detection/crop/rotate pipeline. Off means one model call per request instead
    # of three, and nothing written to detection_output_dir. A request may override this either
    # way with the X-Detection header.
    detection_enabled: bool = True
    detection_output_dir: str = "detections"
    detection_pdf_dpi: int = 200
    detection_crop_padding_px: int = 10
    # Equal on every side: orientation is only known after cropping, so there is no way to tell
    # in advance which edge will end up as the document's top and deserve the extra allowance.
    # Small, because edge refinement puts the corners on the document's real border — the old
    # generous margin only existed to absorb inaccurate corners, and showed up as background.
    detection_crop_padding_ratio: float = 0.015
    detection_crop_top_padding_ratio: float = 0.015
    detection_rotation_min_confidence: float = 0.7
    detection_crop_target_long_edge: int = 1600
    detection_crop_sharpen_amount: float = 0.6
    detection_crop_contrast_clip: float = 1.5
    # A fourth model call per document (on top of extraction, and crop+rotation when detection is
    # on): Gemini looks at the document again alongside its own extracted fields and judges what a
    # field-by-field rule can't - MRZ-to-printed-data consistency, front/back agreement, and other
    # internal contradictions. Same override pattern as detection_enabled: the X-Validation header
    # overrides this per request either way.
    validation_enabled: bool = True
    document_retention_seconds: int = 0
    extracted_data_retention_seconds: int = 0
    api_response_retention_seconds: int = 0
    log_retention_days: int = 30
    # Where the /ocr/leasing endpoints write each request's originals, crops and merged PDFs,
    # one folder per document_refnumber.
    leasing_output_dir: str = "postman_output"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()