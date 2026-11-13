"""
QUESTOHIVE PAST QUESTION PROCESSOR

Processing pipeline:

1. DeepSeek Vision:
   Extract course title and course code from the uploaded image.

2. Jev Stage 1:
   Identify university, faculty, academic session,
   semester and academic level.

3. University identification:
   First batch: maximum 255 universities.
   Remaining batch(es): searched only when the university
   is not found in the first batch.

4. Jev Stage 2:
   Evaluate every department independently using Noul.

5. Final output:
   Print extracted metadata and processing-time breakdown.

Performance:
   - DeepSeek and Jev Stage 1 run concurrently.
   - University fallback and department evaluation run
     concurrently when both are required.
   - Local file operations run outside the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os

from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    Image,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "image.png"

FACULTY_PATH = BASE_DIR / "faculties.json"

DEPARTMENT_PATH = BASE_DIR / "departments.json"

UNIVERSITY_PATH = BASE_DIR / "universities.json"

ENV_PATH = BASE_DIR / ".env"


# Load environment variables from the project directory.
load_dotenv(ENV_PATH)

JEV_API_KEY = os.getenv("JEV_API_KEY")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# Models
JEV_MODEL = "jev-latest"

DEEPSEEK_MODEL = "deepseek-flash"


# University configuration
MAX_UNIVERSITY_OPTIONS = 255

UNIVERSITY_MATCH_THRESHOLD = 0.5


# Department configuration
DEPARTMENT_YES_THRESHOLD = 0.5


# General classifications
ALL_OPTION = "ALL"

NOT_VISIBLE = "NOT_VISIBLE"

NOT_IN_REFERENCE = "NOT_IN_REFERENCE"

REVIEW_REQUIRED = "REVIEW_REQUIRED"


# API configuration
REQUEST_TIMEOUT_SECONDS = 60.0

DEEPSEEK_MAX_IMAGE_BYTES = 32 * 1024 * 1024


# Academic sessions
ACADEMIC_SESSIONS = [
    f"{year}/{year + 1}"
    for year in range(2013, 2027)
]

ACADEMIC_SESSIONS.append(NOT_VISIBLE)


# Academic semesters
SEMESTERS = [
    "First Semester",
    "Second Semester",
    "Third Semester",
    NOT_VISIBLE,
]


# Academic levels
LEVELS = [
    "100 Level",
    "200 Level",
    "300 Level",
    "400 Level",
    "500 Level",
    "600 Level",
    "700 Level",
    NOT_VISIBLE,
]


SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def build_criteria(options: list[str]) -> dict:

    if not options:

        raise ValueError(
            "Choice requires at least one option."
        )

    if len(options) > MAX_UNIVERSITY_OPTIONS:

        raise ValueError(
            f"Choice has {len(options)} options. "
            f"Maximum allowed: {MAX_UNIVERSITY_OPTIONS}"
        )

    return {
        option: None
        for option in options
    }


def load_json(path: Path) -> dict:

    if not path.is_file():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:

        data = json.load(file)

    if not isinstance(data, dict):

        raise ValueError(
            f"Expected a JSON object in {path.name}"
        )

    return data


def split_into_batches(
    items: list[str],
    batch_size: int = MAX_UNIVERSITY_OPTIONS,
) -> list[list[str]]:

    if batch_size < 1 or batch_size > MAX_UNIVERSITY_OPTIONS:

        raise ValueError(
            f"Batch size must be between 1 and "
            f"{MAX_UNIVERSITY_OPTIONS}"
        )

    return [
        items[index:index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def selected_choice_probability(answer: Any):

    probabilities = getattr(
        answer,
        "probabilities",
        None
    ) or {}

    probability = probabilities.get(
        answer.choice
    )

    if probability is None:
        return None

    return float(probability)


# ============================================================
# 3. LOAD REFERENCE DATA
# ============================================================

def load_faculties() -> list[str]:

    data = load_json(FACULTY_PATH)

    groups = data.get("faculties")

    if not isinstance(groups, dict):

        raise ValueError(
            "faculties.json must contain a 'faculties' object."
        )

    faculties = []

    for group_name, group in groups.items():

        if not isinstance(group, list):

            raise ValueError(
                f"Faculty group '{group_name}' must be a list."
            )

        faculties.extend(
            name.strip()
            for name in group
            if isinstance(name, str) and name.strip()
        )

    faculties.extend([
        ALL_OPTION,
        NOT_VISIBLE,
    ])

    return list(dict.fromkeys(faculties))


def load_departments() -> dict[str, list[str]]:

    data = load_json(DEPARTMENT_PATH)

    raw_departments = data.get("departments")

    if not isinstance(raw_departments, dict):

        raise ValueError(
            "departments.json must contain a 'departments' object."
        )

    departments = {}

    for faculty, names in raw_departments.items():

        if not isinstance(names, list):

            raise ValueError(
                f"Departments for '{faculty}' must be a list."
            )

        cleaned = [
            name.strip()
            for name in names
            if isinstance(name, str) and name.strip()
        ]

        departments[faculty] = list(
            dict.fromkeys(cleaned)
        )

    return departments


def load_universities():

    """
    Load federal, state and private universities.

    Returns:
        university_names:
            Ordered list of unique universities.

        university_types:
            Mapping between university name and classification.
    """

    data = load_json(UNIVERSITY_PATH)

    groups = data.get("universities")

    if not isinstance(groups, dict):

        raise ValueError(
            "universities.json must contain a 'universities' object."
        )

    university_names = []

    university_types = {}

    for group_name in ("federal", "state", "private"):

        names = groups.get(group_name, [])

        if not isinstance(names, list):

            raise ValueError(
                f"The '{group_name}' university entry must be a list."
            )

        for raw_name in names:

            if not isinstance(raw_name, str):
                continue

            name = raw_name.strip()

            if not name:
                continue

            if name not in university_types:

                university_names.append(name)

                university_types[name] = group_name.upper()

    if not university_names:

        raise ValueError(
            "No universities were found in universities.json."
        )

    declared_total = data.get("total")

    if (
        isinstance(declared_total, int)
        and declared_total != len(university_names)
    ):

        print(
            f"[Warning] JSON declares {declared_total} universities, "
            f"but {len(university_names)} unique universities "
            "were loaded."
        )

    return university_names, university_types


# ============================================================
# 4. TERMINAL DISPLAY UTILITIES
# ============================================================

def print_timing(label: str, seconds: float):

    print(
        f"{label:<56} {seconds:>9.3f} seconds"
    )


def print_choice_result(label: str, answer: Any):

    print(f"\n{label}: {answer.choice}")

    print(
        f"Confidence: {float(answer.confidence):.2%}"
    )


def print_top_matches(answer: Any, limit: int = 5):

    probabilities = getattr(
        answer,
        "probabilities",
        None
    ) or {}

    if not probabilities:
        return

    ranked = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nTop matches:")

    for name, probability in ranked[:limit]:

        print(
            f"  {name}: {float(probability):.2%}"
        )


# ============================================================
# 5. DEEPSEEK VISION IMAGE PREPARATION
# ============================================================

def read_and_encode_image(image_path: Path):

    image_bytes = image_path.read_bytes()

    image_size = len(image_bytes)

    if image_size > DEEPSEEK_MAX_IMAGE_BYTES:

        raise ValueError(
            "Image exceeds DeepSeek's 32 MiB inline-image limit. "
            "Use the DeepSeek Files API for larger images."
        )

    mime_type, _ = mimetypes.guess_type(
        str(image_path)
    )

    if not mime_type or not mime_type.startswith("image/"):

        mime_type = "image/png"

    encoded = base64.b64encode(
        image_bytes
    ).decode("ascii")

    data_url = (
        f"data:{mime_type};base64,{encoded}"
    )

    return data_url, mime_type, image_size


# ============================================================
# 6. DEEPSEEK VISION: COURSE TITLE AND CODE
# ============================================================

async def extract_course_and_code(
    deepseek_client: AsyncOpenAI,
    image_path: Path,
    timings: dict,
):

    print("\n[DeepSeek] Preparing examination image...")

    preparation_started = perf_counter()

    data_url, mime_type, image_size = await asyncio.to_thread(
        read_and_encode_image,
        image_path
    )

    timings["DeepSeek image preparation"] = (
        perf_counter() - preparation_started
    )

    print(
        f"[DeepSeek] Image size: {image_size:,} bytes"
    )

    print(
        f"[DeepSeek] Image format: {mime_type}"
    )

    request_started = perf_counter()

    try:

        response = await deepseek_client.chat.completions.create(

            model=DEEPSEEK_MODEL,

            messages=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",

                            "text": (

                                "Carefully inspect the uploaded "
                                "examination-paper image. "

                                "Extract the course title and "
                                "course code exactly as printed. "

                                "Return only the course title "
                                "and course code. "

                                "Do not explain, guess, correct "
                                "or add extra information. "

                                "Preserve the original wording "
                                "and capitalization. "

                                "If neither the course title nor "
                                "course code is visible, return "
                                "exactly NOT_VISIBLE. "

                                "If only one is visible, return "
                                "only the visible information."
                            )
                        },

                        {
                            "type": "image_url",

                            "image_url": {

                                "url": data_url,

                                "detail": "original"
                            }
                        }
                    ]
                }
            ]
        )

    finally:

        timings["DeepSeek Vision API"] = (
            perf_counter() - request_started
        )

    if not response.choices:

        return NOT_VISIBLE

    extracted_text = response.choices[0].message.content

    if not isinstance(extracted_text, str):

        return NOT_VISIBLE

    if not extracted_text.strip():

        return NOT_VISIBLE

    return extracted_text.strip()


# ============================================================
# 7. JEV STAGE 1
# UNIVERSITY BATCH 1 + FACULTY + SESSION + SEMESTER + LEVEL
# ============================================================

async def identify_first_stage(
    jev_client: AsyncTypeSafeClient,
    screenshot: Image,
    faculties: list[str],
    university_batch: list[str],
    timings: dict,
):

    if len(university_batch) > MAX_UNIVERSITY_OPTIONS:

        raise ValueError(
            "First university batch exceeds Jev's 255-option limit."
        )

    university_context = (

        "Candidate universities for this batch:\n"

        + json.dumps(
            university_batch,
            ensure_ascii=False
        )
    )

    request_started = perf_counter()

    try:

        response = await jev_client.system_one(

            state=[

                (
                    "Inspect the attached examination-paper image. "

                    "Use the actual image as the evidence source. "

                    "Identify the university and academic metadata "
                    "supported by the visible document."
                ),

                university_context,

                screenshot
            ],

            questions={

                # ------------------------------------
                # UNIVERSITY IDENTIFICATION
                # ------------------------------------

                "university": Choice(

                    instructions=(

                        "Identify the university represented by "
                        "the uploaded examination-paper image. "

                        "Examine the university name, logo, crest, "
                        "letterhead, watermark or other distinctive "
                        "institutional identifiers. "

                        "Select the exact matching university "
                        "from the available options. "

                        "Do not select based on incidental text, "
                        "student names, locations or assumptions."
                    ),

                    criteria=build_criteria(
                        university_batch
                    )
                ),

                # ------------------------------------
                # UNIVERSITY BATCH MEMBERSHIP
                # ------------------------------------

                "university_in_batch": Noul(

                    instructions=(

                        "Does the uploaded examination image "
                        "represent one of the universities "
                        "listed in the candidate university "
                        "batch provided in the shared state? "

                        "Answer YES only when the visible "
                        "university name, logo, crest, "
                        "letterhead or another distinctive "
                        "institutional identifier matches "
                        "a university in this batch. "

                        "Answer NO when the correct university "
                        "is not in this batch or there is no "
                        "reliable evidence of a match. "

                        "Do not count incidental mentions."
                    )
                ),

                # ------------------------------------
                # FACULTY
                # ------------------------------------

                "faculty": Choice(

                    instructions=(

                        "Identify the faculty responsible for "
                        "this past question using the uploaded image. "

                        "Select ALL only when the document "
                        "clearly indicates that the course is "
                        "university-wide and shared across faculties. "

                        "Do not select ALL merely because "
                        "no department is printed. "

                        "Select NOT_VISIBLE if the faculty "
                        "cannot be reliably identified."
                    ),

                    criteria=build_criteria(faculties)
                ),

                # ------------------------------------
                # ACADEMIC SESSION
                # ------------------------------------

                "session": Choice(

                    instructions=(

                        "Identify the academic session printed "
                        "on this examination paper. "

                        "Do not infer it from the current year "
                        "or unrelated numbers. "

                        "Select NOT_VISIBLE if the session is "
                        "absent or unreadable."
                    ),

                    criteria=build_criteria(
                        ACADEMIC_SESSIONS
                    )
                ),

                # ------------------------------------
                # SEMESTER
                # ------------------------------------

                "semester": Choice(

                    instructions=(

                        "Identify the semester printed on "
                        "the uploaded examination paper. "

                        "Select NOT_VISIBLE if it is absent "
                        "or cannot be read."
                    ),

                    criteria=build_criteria(SEMESTERS)
                ),

                # ------------------------------------
                # ACADEMIC LEVEL
                # ------------------------------------

                "level": Choice(

                    instructions=(

                        "Identify the academic level shown "
                        "on the uploaded image. "

                        "Use an explicit level label where present. "

                        "A three-digit course code can also "
                        "indicate the academic level through "
                        "its first digit. "

                        "For example: "

                        "GET 301 indicates 300 Level. "

                        "MEE 210 indicates 200 Level. "

                        "Do not use unrelated numbers. "

                        "Select NOT_VISIBLE if the level "
                        "cannot be established."
                    ),

                    criteria=build_criteria(LEVELS)
                )
            }
        )

    finally:

        timings[
            "Jev Stage 1 (university batch 1 + metadata)"
        ] = perf_counter() - request_started

    return {

        "university_answer": response.choices["university"],

        "university_batch_probability": float(
            response.nouls["university_in_batch"].noul
        ),

        "faculty_answer": response.choices["faculty"],

        "session_answer": response.choices["session"],

        "semester_answer": response.choices["semester"],

        "level_answer": response.choices["level"],
    }


# ============================================================
# 8. UNIVERSITY FALLBACK SEARCH
# ============================================================

async def search_university_batch(
    jev_client: AsyncTypeSafeClient,
    screenshot: Image,
    university_batch: list[str],
    batch_number: int,
    timings: dict,
):

    if not university_batch:
        raise ValueError("University batch is empty.")

    if len(university_batch) > MAX_UNIVERSITY_OPTIONS:

        raise ValueError(
            f"University batch {batch_number} exceeds 255 options."
        )

    print(
        f"\nSearching university batch {batch_number}..."
    )

    batch_context = (

        f"Candidate universities for batch {batch_number}:\n"

        + json.dumps(
            university_batch,
            ensure_ascii=False
        )
    )

    request_started = perf_counter()

    try:

        response = await jev_client.system_one(

            state=[

                (
                    "Identify the university represented by "
                    "the attached examination-paper image. "

                    "The previous university batch did not "
                    "contain a sufficiently supported match. "

                    "Use the uploaded image as evidence."
                ),

                batch_context,

                screenshot
            ],

            questions={

                "university": Choice(

                    instructions=(

                        "Identify the university represented "
                        "by the uploaded examination image. "

                        "Compare the visible university name, "
                        "logo, crest, letterhead, watermark "
                        "or other distinctive identifiers "
                        "against the available options. "

                        "Select the exact matching university. "
                        "Do not rely on incidental mentions."
                    ),

                    criteria=build_criteria(
                        university_batch
                    )
                ),

                "university_in_batch": Noul(

                    instructions=(

                        "Does the uploaded examination image "
                        "represent one of the universities "
                        "listed in this batch? "

                        "Answer YES only when the visible "
                        "institutional evidence supports a "
                        "match to a candidate in this batch. "

                        "Answer NO if the university is not "
                        "in this list or there is no reliable "
                        "evidence of a match."
                    )
                )
            }
        )

    finally:

        timings[
            f"Jev university batch {batch_number}"
        ] = perf_counter() - request_started

    answer = response.choices["university"]

    membership_probability = float(
        response.nouls["university_in_batch"].noul
    )

    selected_probability = selected_choice_probability(
        answer
    )

    print_top_matches(answer)

    print(
        f"Batch {batch_number} membership probability: "
        f"{membership_probability:.2%}"
    )

    if selected_probability is not None:

        print(
            f"Selected university probability: "
            f"{selected_probability:.2%}"
        )

    return {

        "answer": answer,

        "membership_probability": membership_probability,

        "selected_probability": selected_probability,

        "batch_number": batch_number,
    }


async def search_remaining_universities(
    jev_client: AsyncTypeSafeClient,
    screenshot: Image,
    remaining_batches: list[list[str]],
    university_types: dict[str, str],
    timings: dict,
):

    membership_results = {}

    for index, batch in enumerate(
        remaining_batches,
        start=2
    ):

        batch_result = await search_university_batch(

            jev_client,

            screenshot,

            batch,

            index,

            timings
        )

        membership_probability = (
            batch_result["membership_probability"]
        )

        membership_results[f"batch_{index}"] = (
            membership_probability
        )

        # Noul determines whether the university belongs to this batch.
        # Do not apply the 0.5 threshold to the Choice probability.
        if membership_probability > UNIVERSITY_MATCH_THRESHOLD:

            answer = batch_result["answer"]

            university = answer.choice

            return {

                "university": university,

                "university_type": university_types.get(
                    university,
                    "UNKNOWN"
                ),

                "university_status": f"MATCHED_IN_BATCH_{index}",

                "university_batch_used": index,

                "university_choice_probability": (
                    batch_result["selected_probability"]
                ),

                "university_batch_membership_probabilities": (
                    membership_results
                )
            }

    return {

        "university": NOT_IN_REFERENCE,

        "university_type": "UNKNOWN",

        "university_status": "NOT_FOUND_IN_REFERENCE",

        "university_batch_used": None,

        "university_choice_probability": None,

        "university_batch_membership_probabilities": (
            membership_results
        )
    }


# ============================================================
# 9. RESOLVE UNIVERSITY FROM FIRST BATCH
# ============================================================

def resolve_first_batch(
    first_stage: dict,
    university_types: dict[str, str],
):

    membership_probability = (
        first_stage["university_batch_probability"]
    )

    if membership_probability <= UNIVERSITY_MATCH_THRESHOLD:

        return None

    answer = first_stage["university_answer"]

    university = answer.choice

    return {

        "university": university,

        "university_type": university_types.get(
            university,
            "UNKNOWN"
        ),

        "university_status": "MATCHED_IN_BATCH_1",

        "university_batch_used": 1,

        "university_choice_probability": (
            selected_choice_probability(answer)
        ),

        "university_batch_membership_probabilities": {

            "batch_1": membership_probability
        }
    }


# ============================================================
# 10. JEV STAGE 2: DEPARTMENT IDENTIFICATION USING NOUL
# ============================================================

async def identify_departments(
    jev_client: AsyncTypeSafeClient,
    screenshot: Image,
    faculty: str,
    level: str,
    available_departments: list[str],
    timings: dict,
):

    if not available_departments:

        raise ValueError(
            f"No departments found for faculty '{faculty}'."
        )

    questions_started = perf_counter()

    questions = {

        # ----------------------------------------
        # CHECK WHETHER A DEPARTMENT WAS SPECIFIED
        # ----------------------------------------

        "no_department_specified": Noul(

            instructions=(

                "Inspect the uploaded past-question image. "

                "Does the examination paper leave the "
                "intended academic department unspecified? "

                "Answer YES if no particular department "
                "is named or designated as the intended "
                "audience. "

                "Answer NO if one or more specific "
                "departments are clearly identified. "

                "Do not count incidental mentions of "
                "department names. "

                "IMPORTANT: If no department is specified, "
                "the question is applicable to EVERY "
                "department within the identified faculty. "

                "If a department field exists but its "
                "contents are unreadable, do not treat "
                "that as an intentionally omitted department."
            )
        )
    }

    # ----------------------------------------
    # CREATE ONE NOUL QUESTION PER DEPARTMENT
    # ----------------------------------------

    for index, department in enumerate(
        available_departments
    ):

        questions[f"department_{index}"] = Noul(

            instructions=(

                "Using the uploaded examination image, "
                "determine whether this paper is intended "
                f"for students in the '{department}' "
                f"department of '{faculty}'. "

                f"The identified academic level is '{level}'. "

                "Answer YES if the document explicitly "
                "identifies this department as an intended "
                "audience. This includes papers shared "
                "between multiple departments. "

                "Answer NO if the document identifies "
                "other specific departments and this "
                "department is not among them. "

                "IMPORTANT QUESTOHIVE RULE: "

                "If the paper does not specify any "
                "department, it is applicable to EVERY "
                "department within the identified faculty. "

                "Therefore, answer YES for this department "
                "in that situation. "

                "Do not mistake incidental mentions "
                "of department names for the intended audience. "

                "Base the decision on the uploaded image, "
                "not assumptions about which departments "
                "usually study this course."
            )
        )

    timings["Noul question construction"] = (
        perf_counter() - questions_started
    )

    # ----------------------------------------
    # SEND ALL DEPARTMENT QUESTIONS TO JEV
    # ----------------------------------------

    request_started = perf_counter()

    try:

        response = await jev_client.system_one(

            state=[

                (
                    "Inspect the attached past-question image "
                    "to determine all departments for which "
                    "the paper is intended. "

                    f"Identified faculty: {faculty}. "

                    f"Identified academic level: {level}. "

                    "Evaluate every department independently "
                    "using the uploaded image."
                ),

                screenshot
            ],

            questions=questions
        )

    finally:

        timings["Jev Stage 2 department Noul"] = (
            perf_counter() - request_started
        )

    # ----------------------------------------
    # EXTRACT NOUL PROBABILITIES
    # ----------------------------------------

    no_department_probability = float(

        response.nouls[
            "no_department_specified"
        ].noul
    )

    department_probabilities = {}

    selected_departments = []

    print("\nDEPARTMENT IDENTIFICATION RESULTS")

    print("-" * 65)

    print(
        "Probability that no department is specified: "
        f"{no_department_probability:.2%}"
    )

    # ----------------------------------------
    # EVALUATE EVERY DEPARTMENT
    # ----------------------------------------

    for index, department in enumerate(
        available_departments
    ):

        probability = float(

            response.nouls[
                f"department_{index}"
            ].noul
        )

        department_probabilities[department] = probability

        included = (
            probability > DEPARTMENT_YES_THRESHOLD
        )

        print(

            f"{department}: {probability:.2%} "
            f"{'-> INCLUDED' if included else '-> EXCLUDED'}"
        )

        if included:

            selected_departments.append(
                department
            )

    # ----------------------------------------
    # APPLY QUESTOHIVE DISTRIBUTION RULE
    # ----------------------------------------

    if no_department_probability > DEPARTMENT_YES_THRESHOLD:

        selected_departments = list(
            available_departments
        )

        distribution = "ALL_DEPARTMENTS_IN_FACULTY"

        print(
            "\nNo department was specified."
        )

        print(
            f"All departments in {faculty} will receive "
            f"this question at {level}."
        )

    else:

        if not selected_departments:

            distribution = REVIEW_REQUIRED

        elif len(selected_departments) == len(
            available_departments
        ):

            distribution = "ALL_DEPARTMENTS_IN_FACULTY"

        else:

            distribution = "SELECTED_DEPARTMENTS"

    return {

        "departments": selected_departments,

        "department_probabilities": department_probabilities,

        "distribution": distribution,

        "no_department_specified_probability": (
            no_department_probability
        )
    }


# ============================================================
# 11. MAIN ASYNCHRONOUS PROCESSING FUNCTION
# ============================================================

async def process_past_question():

    overall_started = perf_counter()

    timings = {}

    # ========================================================
    # STEP 1: VALIDATE CONFIGURATION
    # ========================================================

    started = perf_counter()

    if not JEV_API_KEY:

        raise ValueError(
            "JEV_API_KEY was not found in .env"
        )

    if not DEEPSEEK_API_KEY:

        raise ValueError(
            "DEEPSEEK_API_KEY was not found in .env"
        )

    if not IMAGE_PATH.is_file():

        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if IMAGE_PATH.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:

        raise ValueError(
            "Unsupported image format. "
            "Use PNG, JPG, JPEG, WEBP or GIF."
        )

    timings["Configuration validation"] = (
        perf_counter() - started
    )

    print("\n" + "=" * 75)

    print(
        "QUESTOHIVE PAST QUESTION ANALYSIS"
    )

    print(
        "ASYNCHRONOUS PROCESSING PIPELINE"
    )

    print("=" * 75)

    print(f"\nImage: {IMAGE_PATH.name}")

    # ========================================================
    # STEP 2: LOAD REFERENCE JSON FILES CONCURRENTLY
    # ========================================================

    started = perf_counter()

    faculties, departments_by_faculty, university_data = (
        await asyncio.gather(

            asyncio.to_thread(
                load_faculties
            ),

            asyncio.to_thread(
                load_departments
            ),

            asyncio.to_thread(
                load_universities
            )
        )
    )

    universities, university_types = university_data

    timings["Reference JSON loading"] = (
        perf_counter() - started
    )

    # ========================================================
    # STEP 3: CREATE UNIVERSITY BATCHES
    # ========================================================

    university_batches = split_into_batches(
        universities,
        MAX_UNIVERSITY_OPTIONS
    )

    print(
        f"\nTotal universities: {len(universities)}"
    )

    for index, batch in enumerate(
        university_batches,
        start=1
    ):

        print(
            f"University batch {index}: {len(batch)} options"
        )

    # For your current dataset of 328 universities,
    # this should display:
    #
    # University batch 1: 255 options
    # University batch 2: 73 options
    #
    # If your actual JSON contains a different number,
    # the batches are generated automatically.

    # ========================================================
    # STEP 4: PREPARE IMAGE FOR JEV
    # ========================================================

    started = perf_counter()

    screenshot = await asyncio.to_thread(
        Image,
        IMAGE_PATH
    )

    timings["Jev image preparation"] = (
        perf_counter() - started
    )

    # ========================================================
    # STEP 5: INITIALIZE ASYNC CLIENTS
    # ========================================================

    jev_init_started = perf_counter()

    async with AsyncTypeSafeClient(

        api_key=JEV_API_KEY,

        model=JEV_MODEL,

        timeout=REQUEST_TIMEOUT_SECONDS,

        headers={

            "TypeSafe-Danger-Zone-Images": "true"

        }

    ) as jev_client:

        timings["Jev client initialization"] = (
            perf_counter() - jev_init_started
        )

        deepseek_init_started = perf_counter()

        async with AsyncOpenAI(

            api_key=DEEPSEEK_API_KEY,

            base_url="https://api.deepseek.com",

            timeout=REQUEST_TIMEOUT_SECONDS,

            max_retries=1

        ) as deepseek_client:

            timings["DeepSeek client initialization"] = (
                perf_counter() - deepseek_init_started
            )

            # ====================================================
            # STEP 6: RUN DEEPSEEK AND JEV STAGE 1 CONCURRENTLY
            # ====================================================

            print(
                "\nSTAGE 0 + STAGE 1"
            )

            print(
                "Running DeepSeek Vision and Jev concurrently..."
            )

            parallel_started = perf_counter()

            deepseek_task = asyncio.create_task(

                extract_course_and_code(

                    deepseek_client,

                    IMAGE_PATH,

                    timings
                )
            )

            jev_stage1_task = asyncio.create_task(

                identify_first_stage(

                    jev_client,

                    screenshot,

                    faculties,

                    university_batches[0],

                    timings
                )
            )

            # Wait for both independent requests.

            stage1_results = await asyncio.gather(

                deepseek_task,

                jev_stage1_task,

                return_exceptions=True
            )

            timings["DeepSeek + Jev Stage 1 parallel wall time"] = (
                perf_counter() - parallel_started
            )

            for task_result in stage1_results:

                if isinstance(
                    task_result,
                    BaseException
                ):

                    raise task_result

            course_and_code = stage1_results[0]

            first_stage = stage1_results[1]

            university_answer = (
                first_stage["university_answer"]
            )

            university_batch_probability = (
                first_stage["university_batch_probability"]
            )

            faculty_answer = (
                first_stage["faculty_answer"]
            )

            session_answer = (
                first_stage["session_answer"]
            )

            semester_answer = (
                first_stage["semester_answer"]
            )

            level_answer = (
                first_stage["level_answer"]
            )

            faculty = faculty_answer.choice

            session = session_answer.choice

            semester = semester_answer.choice

            level = level_answer.choice

            # ====================================================
            # STEP 7: DISPLAY STAGE 0 AND STAGE 1
            # ====================================================

            print("\nSTAGE 0: DEEPSEEK VISION")

            print("-" * 65)

            print(
                f"Course title / code: {course_and_code}"
            )

            print("\nSTAGE 1: JEV ACADEMIC METADATA")

            print("-" * 65)

            print_choice_result(
                "Faculty",
                faculty_answer
            )

            print_choice_result(
                "Academic session",
                session_answer
            )

            print_choice_result(
                "Semester",
                semester_answer
            )

            print_choice_result(
                "Academic level",
                level_answer
            )

            # ====================================================
            # STEP 8: RESOLVE UNIVERSITY BATCH 1
            # ====================================================

            print("\nUNIVERSITY IDENTIFICATION: BATCH 1")

            print("-" * 65)

            print(
                f"Universities evaluated: "
                f"{len(university_batches[0])}"
            )

            print_top_matches(university_answer)

            print(
                "Probability university belongs to batch 1: "
                f"{university_batch_probability:.2%}"
            )

            if university_batch_probability > UNIVERSITY_MATCH_THRESHOLD:

                university = university_answer.choice

                university_result = {

                    "university": university,

                    "university_type": university_types.get(
                        university,
                        "UNKNOWN"
                    ),

                    "university_status": "MATCHED_IN_BATCH_1",

                    "university_batch_used": 1,

                    "university_choice_probability": (
                        selected_choice_probability(
                            university_answer
                        )
                    ),

                    "university_batch_membership_probabilities": {

                        "batch_1": university_batch_probability
                    }
                }

                print(
                    f"\nUniversity identified: {university}"
                )

            else:

                university_result = None

                print(
                    "\nUniversity was not identified in batch 1."
                )

                print(
                    "Checking the remaining university batch(es)..."
                )

            # ====================================================
            # STEP 9: PREPARE POST-STAGE-1 TASKS
            # ====================================================

            pending_tasks = {}

            # ----------------------------------------
            # UNIVERSITY FALLBACK
            # ----------------------------------------

            if university_result is None:

                remaining_batches = university_batches[1:]

                if remaining_batches:

                    pending_tasks["university"] = (
                        search_remaining_universities(

                            jev_client,

                            screenshot,

                            remaining_batches,

                            university_types,

                            timings
                        )
                    )

                else:

                    university_result = {

                        "university": NOT_IN_REFERENCE,

                        "university_type": "UNKNOWN",

                        "university_status": "NOT_FOUND_IN_REFERENCE",

                        "university_batch_used": None,

                        "university_choice_probability": None,

                        "university_batch_membership_probabilities": {}
                    }

            # ----------------------------------------
            # DEPARTMENT IDENTIFICATION
            # ----------------------------------------

            if faculty not in (
                ALL_OPTION,
                NOT_VISIBLE
            ):

                available_departments = (
                    departments_by_faculty.get(
                        faculty,
                        []
                    )
                )

                if not available_departments:

                    raise ValueError(
                        f"No departments found for faculty '{faculty}'. "
                        "Check departments.json."
                    )

                pending_tasks["departments"] = (
                    identify_departments(

                        jev_client,

                        screenshot,

                        faculty,

                        level,

                        available_departments,

                        timings
                    )
                )

            # ====================================================
            # STEP 10: RUN INDEPENDENT REQUESTS CONCURRENTLY
            # ====================================================

            # Once Stage 1 has completed, university fallback
            # and department identification are independent.
            #
            # They can therefore run concurrently.

            department_result = None

            if pending_tasks:

                print(
                    "\nSTAGE 2: Running department evaluation "
                    "and university fallback concurrently..."
                )

                stage2_started = perf_counter()

                task_names = list(
                    pending_tasks.keys()
                )

                task_results = await asyncio.gather(

                    *pending_tasks.values(),

                    return_exceptions=True
                )

                timings[
                    "Post-Stage-1 concurrent processing wall time"
                ] = perf_counter() - stage2_started

                for name, task_result in zip(
                    task_names,
                    task_results
                ):

                    if isinstance(
                        task_result,
                        BaseException
                    ):

                        raise task_result

                    if name == "university":

                        university_result = task_result

                    elif name == "departments":

                        department_result = task_result

            # ====================================================
            # STEP 11: HANDLE SPECIAL FACULTY CASES
            # ====================================================

            if department_result is None:

                if faculty == ALL_OPTION:

                    department_result = {

                        "departments": [ALL_OPTION],

                        "department_probabilities": {},

                        "distribution": "ALL_DEPARTMENTS",

                        "no_department_specified_probability": None
                    }

                    print(
                        "\nUniversity-wide general course detected."
                    )

                else:

                    department_result = {

                        "departments": [],

                        "department_probabilities": {},

                        "distribution": REVIEW_REQUIRED,

                        "no_department_specified_probability": None
                    }

                    print(
                        "\nFaculty could not be identified. "
                        "Department identification skipped."
                    )

            if university_result is None:

                raise RuntimeError(
                    "University identification did not produce a result."
                )

            # ====================================================
            # STEP 12: FINAL METADATA
            # ====================================================

            result = {

                "document_type": "past_question",

                "university": university_result["university"],

                "university_type": university_result["university_type"],

                "university_status": university_result["university_status"],

                "university_batch_used": university_result["university_batch_used"],

                "university_batch_membership_probabilities": (
                    university_result[
                        "university_batch_membership_probabilities"
                    ]
                ),

                "university_choice_probability": (
                    university_result["university_choice_probability"]
                ),

                "course_and_code": course_and_code,

                "faculty": faculty,

                "departments": department_result["departments"],

                "academic_session": session,

                "semester": semester,

                "level": level,

                "distribution": department_result["distribution"],

                "department_probabilities": (
                    department_result["department_probabilities"]
                ),

                "no_department_specified_probability": (
                    department_result[
                        "no_department_specified_probability"
                    ]
                ),

                "confidence": {

                    "faculty": float(
                        faculty_answer.confidence
                    ),

                    "session": float(
                        session_answer.confidence
                    ),

                    "semester": float(
                        semester_answer.confidence
                    ),

                    "level": float(
                        level_answer.confidence
                    )
                }
            }

    # ========================================================
    # STEP 13: PRINT FINAL METADATA
    # ========================================================

    print("\n" + "=" * 75)

    print(
        "FINAL QUESTOHIVE METADATA"
    )

    print("=" * 75)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    if result["distribution"] == REVIEW_REQUIRED:

        print(
            "\nManual review required."
        )

    else:

        print(
            "\nPast-question metadata extraction completed."
        )

    # ========================================================
    # STEP 14: PROCESSING TIME REPORT
    # ========================================================

    overall_seconds = (
        perf_counter() - overall_started
    )

    jev_api_seconds = sum(

        duration

        for label, duration in timings.items()

        if (
            label.startswith("Jev Stage")
            or label.startswith("Jev university batch")
        )
    )

    deepseek_api_seconds = timings.get(
        "DeepSeek Vision API",
        0.0
    )

    print("\n" + "=" * 75)

    print(
        "PROCESSING TIME BREAKDOWN"
    )

    print("=" * 75)

    timing_order = [

        "Configuration validation",

        "Reference JSON loading",

        "Jev image preparation",

        "Jev client initialization",

        "DeepSeek client initialization",

        "DeepSeek image preparation",

        "DeepSeek Vision API",

        "Jev Stage 1 (university batch 1 + metadata)",

        "Jev Stage 2 department Noul",

        "DeepSeek + Jev Stage 1 parallel wall time",

        "Post-Stage-1 concurrent processing wall time"
    ]

    for label in timing_order:

        if label in timings:

            print_timing(
                label,
                timings[label]
            )

    for label, duration in timings.items():

        if label.startswith("Jev university batch "):

            print_timing(
                label,
                duration
            )

    print("-" * 75)

    print_timing(
        "TOTAL JEV API ROUND-TRIP TIME",
        jev_api_seconds
    )

    print_timing(
        "TOTAL DEEPSEEK API ROUND-TRIP TIME",
        deepseek_api_seconds
    )

    print("-" * 75)

    print_timing(
        "OVERALL PIPELINE WALL TIME",
        overall_seconds
    )

    print("=" * 75)

    print(
        "\nTiming information:"
        "\n- API round-trip time includes image upload, network "
        "communication, provider processing and response handling."
        "\n- It does not represent isolated model inference time."
        "\n- Concurrent requests overlap, so the sum of API times "
        "can be greater than the overall pipeline time."
    )

    return result


# ============================================================
# 12. EXECUTION
# ============================================================

async def main():

    await process_past_question()


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print(
            "\nProcessing cancelled by user."
        )

    except Exception as error:

        print(
            f"\nProcessing failed: {error}"
        )

        raise