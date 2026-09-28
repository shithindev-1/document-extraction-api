# Testing through Postman

This is the same OCR pipeline as the existing `ocr_dev_gemini` project - extraction, front/back
pairing and merging, self-verification, cross-verification, cropping, rotation - reachable through
two new endpoints on the same running app:

- **`POST /api/v1/ocr/leasing/upload`** - multipart **form-data**, documents attached directly as files in
  Postman, several at once. **Use this one** - it's what the rest of this guide walks through.
- **`POST /api/v1/ocr/leasing`** - JSON body, documents named by URL instead (the API downloads them
  itself) - covered at the end, in case you need it later.

Both return the exact same response shape. See `app/services/leasing.py` for exactly what either does and
does not change; the original `POST /api/v1/ocr` (the existing multipart endpoint) still works exactly as
before, untouched.

## 1. Set up

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and fill in your own `GEMINI_API_KEY`/`GEMINI_MODEL`. `.env` is never
committed to git.

## 2. Run the server

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Set `REQUIRE_HTTPS=false` in `.env` for local testing, otherwise the API rejects a plain
`http://localhost` request from Postman. The HTTPS and rate-limit checks apply to every route under
`/api/v1/ocr`, leasing included. If `API_KEYS` is set, add an `X-API-Key: <key>` header to every
request (Postman: **Headers** tab).

## 3. Set up the request in Postman (form-data, multiple files)

- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/ocr/leasing/upload`
- **Body tab** → select **form-data**

Add these rows. `file` is type **File** (click the dropdown on the right of the key field to switch
a row from Text to File); everything else is type **Text**:

| Key | Type | Value (example) |
|---|---|---|
| `tenant_type` | Text | `individual` |
| `document_refnumber` | Text | `93ade000-37d4-41f0-bbc7-422dcbe46ad0` |
| `file` | **File** | *(click "Select Files", pick your first document)* |
| `document_id` | Text | `16` |
| `document_type` | Text | `National Id` |
| `document_name` | Text | `Nationalid1` |

Click **Send**. That's it for one document.

### Uploading multiple files at once

Add **one more row for each of `file`, `document_id`, `document_type`, `document_name`**, still
using those exact same key names - Postman lets you repeat a key across several rows, and they are
matched up **by position**: the 2nd `file` row pairs with the 2nd `document_id` row, the 2nd
`document_type` row, and the 2nd `document_name` row, and so on. So for a National ID's front and
back plus a passport, the form-data tab ends up with:

| Key | Type | Value |
|---|---|---|
| `tenant_type` | Text | `individual` |
| `document_refnumber` | Text | `93ade000-...` |
| `file` | File | *National ID front.jpg* |
| `document_id` | Text | `16` |
| `document_type` | Text | `National Id` |
| `document_name` | Text | `National ID Front` |
| `file` | File | *National ID back.jpg* |
| `document_id` | Text | `17` |
| `document_type` | Text | `National Id` |
| `document_name` | Text | `National ID Back` |
| `file` | File | *Passport.jpg* |
| `document_id` | Text | `18` |
| `document_type` | Text | `Passport` |
| `document_name` | Text | `Passport` |

`tenant_type` and `document_refnumber` only ever appear once, at the top - they describe the whole
request, not any one document. Send it: the National ID front and back are detected and merged
automatically (see the existing README's **Endpoint** section for how that works), and all
documents in the request are cross-verified against each other.

If the row order gets mixed up while editing, that's fine as long as each group of 4
(`file`/`document_id`/`document_type`/`document_name`) keeps the same relative order as the others -
Postman sends form-data fields in the order the rows appear, which is what "by position" pairs on.

### Response shape

Success and failure both return this same shape. A successful National ID front + back (merged)
plus a passport:

```json
{
  "status": "success",
  "tenant_type": "individual",
  "document_refnumber": "93ade000-37d4-41f0-bbc7-422dcbe46ad0",
  "form_data": {
    "tenant_name_en": "Ahmed ...",
    "national_id": "784-...",
    "...": "...every other field, null if not read..."
  },
  "missing": ["visa_number", "..."],
  "processing_time": "14.72 sec",
  "documents": [
    {
      "document_type": "national_id",
      "document_type_is_correct": true,
      "merged": true,
      "sources": [
        {"source": "nid_front.jpg", "document_id": "16", "document_name": "NID Front",
         "document_type": "National Id", "pages": [1]},
        {"source": "nid_back.jpg", "document_id": "17", "document_name": "NID Back",
         "document_type": "National Id", "pages": [1]}
      ],
      "error": null
    },
    {
      "document_type": "passport",
      "document_type_is_correct": true,
      "merged": false,
      "sources": [
        {"source": "passport.pdf", "document_id": "18", "document_name": "Passport",
         "document_type": "Passport", "pages": [1, 2]}
      ],
      "error": null
    }
  ],
  "error": null
}
```

A failure - here `/api/v1/ocr/leasing` with an empty `source` URL:

```json
{
  "status": "failed",
  "tenant_type": "individual",
  "document_refnumber": "93ade000-37d4-41f0-bbc7-422dcbe46ad0",
  "form_data": {"tenant_name_en": null, "...": null},
  "missing": ["tenant_name_en", "...every field..."],
  "processing_time": "0 sec",
  "documents": [
    {
      "document_type": "unknown",
      "document_type_is_correct": false,
      "merged": false,
      "sources": [
        {"source": "", "document_id": "16", "document_name": "Nationalid1",
         "document_type": "National Id", "pages": []}
      ],
      "error": "Failed to fetch document from '': Request URL is missing an 'http://' or 'https://' protocol."
    }
  ],
  "error": "Failed to fetch document from '': Request URL is missing an 'http://' or 'https://' protocol."
}
```

- **Every document is checked in full**, even when another one has already failed: validation,
  type check, front/back pairing, self-verification. Cross-verification (do the documents agree
  with each other?) runs only once every document has passed on its own.
- `status` / top-level `error`: whether the *whole request* succeeded. This is all-or-nothing -
  any document's problem empties `form_data` for everyone in the batch. The top-level `error` is
  the **first** failing document's message; check `documents[].error` for all of them.
- `documents[]`: one entry per processed document. A front and back sent as two files and merged
  become **one** entry with `merged: true` and both files under `sources`. They stay as two
  separate entries if the request failed before or during merging.
- `documents[].document_type`: the normalised type (`national_id`, `passport`, ...); `"unknown"`
  if that file never got past download/validation. `sources[].document_type` is what you sent.
- `documents[].document_type_is_correct`: `true` once the file was confirmed to be the type you
  selected; `false` on a type mismatch or if it was never checked.
- `documents[].error`: that document's own problem, or `null` if it passed. A front/back pair
  that doesn't match shows the error on both sides; a cross-verification finding shows on the
  documents it mentions.
- `sources[].pages`: page numbers read from that file - `[1]` for an image, `[1, 2, ...]` for a
  PDF, `[]` if the file was never read.
- Output files are not listed in the response - see the next section for where they are.

## 4. Where the files are saved

Nothing is uploaded anywhere - everything lands under `postman_output/` next to wherever you run
`uvicorn` from, organised like this:

```
postman_output/
  <document_refnumber>/
    <document_id>_<document_name>/
      original_<uploaded filename>         <- the exact file you attached in Postman
      <stem>_<sha8>.json                   <- crop/rotation metadata (only if detection is on)
      <stem>_<sha8>.png                    <- final cropped AND rotated page(s) - one file per side
    merged_<front_id>_<back_id>/           <- only for a front+back pair the pipeline merged
      <stem>_<sha8>.json
      <stem>_<sha8>.png                    <- one per side (front, back)
      <stem>_<sha8>_merged.pdf             <- the two sides combined into one downloadable PDF
```

There is no separate "rotated" file: this pipeline crops and rotates a page in one step, so the
`.png` file *is* the final, upright, cropped result. If a National ID's front and back were sent as
two separate files, only the shared `merged_<front_id>_<back_id>/` folder gets crop/rotate/merge
output - the two `<document_id>_<document_name>/` folders each still keep their own
`original_...` file for traceability, since that is what was actually uploaded for each one.

## 5. Common problems

- **`422 Unprocessable Entity`**: a form-data row is missing or misnamed - check `tenant_type`,
  `document_refnumber` are present, and that `file`/`document_id`/`document_type`/`document_name`
  all have the same number of rows.
- **`400` "Each uploaded file must be sent with its own document_id, document_type, and
  document_name"**: the row counts don't match - e.g. 3 `file` rows but only 2 `document_id` rows.
- **`status: "failed"` with a plain-language `error`**: this is the pipeline itself finding a real
  problem (wrong document type, a missing side with no matching pair, a self- or cross-verification
  finding) - same as the existing endpoint, just reported in this response shape instead.

## Alternative: JSON body with URLs (`POST /api/v1/ocr/leasing`)

If a document is already hosted somewhere reachable and you'd rather send its URL than attach the
file itself, `POST /api/v1/ocr/leasing` takes the same request shape you originally described, as raw JSON
(Content-Type: application/json, set automatically by Postman for "raw" + "JSON" body):

```json
{
  "tenant_type": "individual",
  "document_refnumber": "93ade000-37d4-41f0-bbc7-422dcbe46ad0",
  "sources": [
    {
      "source": "https://<account>.blob.core.windows.net/<container>/ApplicationFiles/LeasingRequestOCR/1/93ade000-37d4-41f0-bbc7-422dcbe46ad0/Nationalid 1.jpg",
      "document_id": "16",
      "document_type": "National Id",
      "document_name": "Nationalid1"
    }
  ]
}
```

Same response shape, same local file layout, same all-or-nothing behaviour - just add more objects
to `sources` for multiple documents. This assumes every `source` URL is directly downloadable with a
plain GET; if yours need a SAS token or auth header, that has to already be part of the URL you send.
