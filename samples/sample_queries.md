# Sample Queries

Sample video: `psychology-lecture.mp4`
(MIT OpenCourseWare - Introduction to Psychology 9.00SC, Lecture 1)

All five queries below are the exact example queries from the assignment spec.
They were tested end-to-end against the full stack and all pass.

---

## Query 1 - Transcribe

> "Transcribe the video."

Expected: Full speech-to-text transcript of the lecture audio.

---

## Query 2 - Generate PowerPoint

> "Create a PowerPoint with the key points discussed in the video."

Expected: A `.pptx` file written to `outputs/`, returned as an artifact path.
Sample output: `samples/sample_output.pptx`

---

## Query 3 - Detect Objects

> "What objects are shown in the video?"

Expected: List of detected objects with confidence scores and timestamps.
Example result: person (0.89), tie (0.67), laptop (0.44)

---

## Query 4 - Detect Graphs

> "Are there any graphs in the video? If yes, describe them."

Expected: Graph/chart detection result plus any OCR-extracted text from frames.
Example result: No graphs detected; 37 text regions found.

---

## Query 5 - Summarize and PDF

> "Summarize our discussion so far and generate a PDF."

Expected: A `.pdf` report written to `outputs/`, returned as an artifact path.
Uses cached transcript from the same session - no re-transcription.
Sample output: `samples/sample_output.pdf`

---

## Clarification example

> "Can you help me with the video?"

Expected: `needs_clarification=true` with options:
- Transcribe the video
- Detect objects in the video
- Detect graphs or text in the video
- Summarize the video
- Generate a PDF report
- Generate a PowerPoint presentation
