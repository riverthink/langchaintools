"You are reviewing a patient's chart to draft a concise clinical summary.
First, retrieve the up-to-date patient record from the hospital EHR using the patient records tool.

Then, review your draft to check for possible documentation errors, such as missing vitals, conflicting meds, or labs that were not pulled.
If you spot an issue, correct it and regenerate the summary.
Continue this process until you are confident the note is accurate or you have attempted three revisions.

You have access to the following tools:
- patient records API — used for retrieving demographics, ward, vitals, and recent notes
- lab results API — used for pulling the latest lab values and timestamps

Structured data injected into the prompt from the tools
------------------------------------------------------
Patient: Jane Doe (MRN 123456) | Ward: Cardiology
Vitals: BP 148/92, HR 88, Temp 37.1 C, SpO2 96%
Meds: Lisinopril 10 mg daily, Atorvastatin 20 mg nightly
Labs: K 4.2 mmol/L (today 08:15), Creatinine 1.1 mg/dL (today 08:15), HbA1c 7.1% (last week)

Draft summary (after tool calling + reflexion cycle):
"67-year-old with hypertension in Cardiology. On Lisinopril and Atorvastatin. Vitals stable (BP 148/92). Labs today show K 4.2, Cr 1.1; HbA1c 7.1% last week. No conflicting meds noted. Continue BP control and monitor renal function."
