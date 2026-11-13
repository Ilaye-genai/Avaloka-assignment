# Avaloka Assignment

This repository contains a Python-based pipeline for extracting academic metadata from examination-paper images. It is designed to read an uploaded past-question image, identify the course information, and infer the associated university, faculty, session, semester, level, and department using external AI APIs.

## What this project does

The main script, `test_jev.py`, performs the following workflow:

- Uses DeepSeek Vision to extract the course title and course code from the image.
- Uses the JEV API to identify:
  - university
  - faculty
  - academic session
  - semester
  - academic level
- Uses faculty and department reference data to infer which departments the past question applies to.
- Prints a final JSON result and a timing breakdown for the processing pipeline.

The project is optimized for speed by running some tasks concurrently, including DeepSeek Vision and JEV metadata extraction.

## Repository structure

- `test_jev.py` — main processing script
- `image.png` — sample examination image used for processing
- `faculties.json` — list of supported faculties
- `departments.json` — faculty-to-department reference mapping
- `universities.json` — university reference dataset
- `LICENSE` — project license
- `.env` — local environment file for API keys (not committed by default)

## Requirements

- Python 3.10+
- `python-dotenv`
- `openai`
- `typesafe_sdk`

Install the required Python packages with:

```bash
pip install python-dotenv openai typesafe_sdk
```

## Configuration

Create a `.env` file in the repository root with your API credentials:

```env
JEV_API_KEY=your_jev_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## Run the project

From the repository root, run:

```bash
python test_jev.py
```

The script will:

1. Validate the environment and input image.
2. Load the university, faculty, and department reference files.
3. Process the image using DeepSeek and JEV.
4. Print the extracted metadata as JSON.
5. Show a processing-time report.

## Notes

- The example image is stored as `image.png` and is used by default.
- The output includes confidence information and metadata about the document.
- If the program cannot confidently identify the university or department, it may flag the document for manual review.

## Example output

The script prints final extracted metadata in JSON format, including fields similar to:

```json
{
  "document_type": "past_question",
  "university": "University Name",
  "faculty": "Engineering",
  "course_and_code": "Course Title / Course Code",
  "academic_session": "2023/2024",
  "semester": "First Semester",
  "level": "300 Level",
  "departments": ["Computer Engineering"],
  "distribution": "SELECTED_DEPARTMENTS"
}
```

## License

This project is licensed under the terms of the included `LICENSE` file.
