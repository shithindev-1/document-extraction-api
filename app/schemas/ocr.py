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

# Only a National ID carries a mandatory front/back pair. Every other document type - passport and
# visa included - is complete as submitted, so a single-page upload of one never raises an
# incomplete error.
FRONT_BACK_TYPES = frozenset({"national_id"})

# Every type the extraction pass is taught to recognise by name, and therefore every type whose
# selection can be checked against what the file actually shows. These are exactly the values the
# client's document-type choices send. A caller may still send some other string - "utility_bill", say -
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
    "cheque",
})

# Other spellings of the VERIFIABLE_TYPES, keyed by their normalised form (lowercase, underscores).
# Applied both to the type a caller declares and to the types the model reports, so "Salary
# Statement", "Emirates ID" or "Cheque Copy" are checked as salary_certificate, national_id and
# cheque rather than slipping through as unknown types.
DOCUMENT_TYPE_ALIASES = {
    "nationalid": "national_id",
    "national_identity": "national_id",
    "emirates_id": "national_id",
    "eid": "national_id",
    "residence_visa": "visa",
    "residency_visa": "visa",
    # British spelling of the same document, which the model uses about as often as the American.
    "trade_licence": "trade_license",
    "business_license": "trade_license",
    "business_licence": "trade_license",
    "ejari": "ejari_certificate",
    "ejari_registration": "ejari_certificate",
    "bank_account_statement": "bank_statement",
    "statement_of_account": "bank_statement",
    "initialapproval": "initial_approval",
    "initial_approval_letter": "initial_approval",
    "salary_certificate_letter": "salary_certificate",
    "salary_cert": "salary_certificate",
    "salary_statement": "salary_certificate",
    "salary_letter": "salary_certificate",
    "salary_slip": "salary_certificate",
    "salary_transfer_letter": "salary_certificate",
    "payslip": "salary_certificate",
    "pay_slip": "salary_certificate",
    "tenancy_agreement": "tenancy_contract",
    "rental_contract": "tenancy_contract",
    "check": "cheque",
    "cheques": "cheque",
    "bank_cheque": "cheque",
    "cheque_copy": "cheque",
    "security_cheque": "cheque",
    "post_dated_cheque": "cheque",
    "postdated_cheque": "cheque",
    "pdc": "cheque",
}


def canonical_document_type(document_type: str) -> str:
    """The one spelling a type is checked under, e.g. "salary_statement" -> "salary_certificate"."""
    return DOCUMENT_TYPE_ALIASES.get(document_type, document_type)

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
