"""Field lists and response models for the /ocr endpoint."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


DATA_FIELDS = [
    "Tenant_Name_En",
    "Tenant_Name_Ar",
    "Date_Of_Birth",
    "Nationality",
    "Gender",
    "Occupation",
    "Employer_Name",
    "National_Id",
    "National_Id_Expiry",
    "Passport_Number",
    "Passport_Issuing_Location",
    "Passport_Expiry",
    "Visa_Number",
    "Visa_Start_Date",
    "Visa_Expiry",
]

# Only these identity documents carry a mandatory front/back pair. Every other document type is
# complete as submitted, so a single-page upload of one must never raise an incomplete error.
FRONT_BACK_TYPES = frozenset({"national_id", "passport", "visa"})

# Every type the extraction pass is taught to recognise by name, and therefore every type whose
# selection can be checked against what the file actually shows. These are exactly the values the
# client's document-type choices send. A caller may still send some other string - "salary_certificate", say -
# which the model has no name for; those are accepted as declared rather than guessed at, because
# rejecting on an observation the model was never asked to make would fail valid uploads.
VERIFIABLE_TYPES = frozenset({
    "national_id",
    "passport",
    "visa",
    "bank_statement",
    "ejari_certificate",
    "trade_license",
    "tenant_form",
    "initial_approval",
    "salary_certificate",
    "tenancy_contract",
})

MISSING_INFO_FIELDS = [
    "tenant_name_en",
    "tenant_name_ar",
    "date_of_birth",
    "nationality",
    "gender",
    "occupation",
    "employer_name",
    "national_id",
    "national_id_expiry",
    "passport_number",
    "passport_issuing_location",
    "passport_expiry",
    "visa_number",
    "visa_start_date",
    "visa_expiry",
]


class OcrData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Tenant_Name_En: str | None = None
    Tenant_Name_Ar: str | None = None
    Date_Of_Birth: str | None = None
    Nationality: str | None = None
    Gender: str | None = None
    Occupation: str | None = None
    Employer_Name: str | None = None
    National_Id: str | None = None
    National_Id_Expiry: str | None = None
    Passport_Number: str | None = None
    Passport_Issuing_Location: str | None = None
    Passport_Expiry: str | None = None
    Visa_Number: str | None = None
    Visa_Start_Date: str | None = None
    Visa_Expiry: str | None = None


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    DocumentName: str
    DocumentFileName: str
    DocumentError: str
    DocumentErrorToShow: str


class OcrResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    userName: str | None = None
    tenantType: str | None = None
    subscriptionId: str | None = None
    ocrReferenceId: str | None = None
    data: OcrData = Field(default_factory=OcrData)
    missingInfo: list[str] = Field(default_factory=list)
    errorInfo: list[ErrorInfo] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_success(self) -> "OcrResponse":
        # success reflects whether any field was actually extracted, not whether the
        # request completed without a transport-level error (that's the HTTP status code's
        # job) — a document that yields nothing usable must report success=false even
        # though the API call itself succeeded.
        self.success = any(value is not None for value in self.data.model_dump().values())
        return self


def empty_response() -> dict[str, Any]:
    return OcrResponse().model_dump()
