import re
from difflib import SequenceMatcher
from typing import Optional


TITLES = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
}

INTERVIEWER_LABELS = {
    "interviewer",
    "hiring manager",
    "hr manager",
    "recruiter",
}

CANDIDATE_LABELS = {
    "candidate",
    "interviewee",
}

METADATA_LABELS = {
    "date",
    "time",
    "location",
    "job title",
    "interview time",
    "interview location",
}

DIALOGUE_STARTERS = {
    "good",
    "hello",
    "hi",
    "thank",
    "thanks",
    "welcome",
    "great",
    "so",
    "okay",
    "can",
    "tell",
    "well",
    "excellent",
    "alright",
    "absolutely",
    "before",
    "lets",
    "let",
    "moving",
    "finally",
}


# ============================================================
# Normalization
# ============================================================

def normalize_speaker_name(name: str) -> str:
    """
    Normalize names and speaker labels.
    """

    name = name.strip().lower()

    # Markdown
    name = re.sub(r"^#+\s*", "", name)
    name = name.replace("**", "")

    # Punctuation
    name = re.sub(r"[^\w\s]", "", name)

    # Whitespace
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def get_name_tokens(name: str) -> list[str]:
    """
    Dr. Rachel Kim -> ["rachel", "kim"]
    """

    normalized = normalize_speaker_name(name)

    return [
        token
        for token in normalized.split()
        if token not in TITLES
    ]


# ============================================================
# Fuzzy person-name matching
# ============================================================

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        a.lower(),
        b.lower(),
    ).ratio()


def token_matches(
    token_a: str,
    token_b: str,
    threshold: float = 0.80,
) -> bool:

    if token_a == token_b:
        return True

    return similarity(
        token_a,
        token_b,
    ) >= threshold


def names_match(
    speaker_name: str,
    person_name: Optional[str],
) -> bool:

    if not person_name:
        return False

    speaker_tokens = get_name_tokens(
        speaker_name
    )

    person_tokens = get_name_tokens(
        person_name
    )

    if not speaker_tokens or not person_tokens:
        return False

    for speaker_token in speaker_tokens:

        matched = any(
            token_matches(
                speaker_token,
                person_token,
            )
            for person_token in person_tokens
        )

        if not matched:
            return False

    return True


# ============================================================
# Label detection
# ============================================================

def is_interviewer_label(
    speaker_name: str,
) -> bool:
    """
    Supports:

        Interviewer
        Interviewer 1
        Interviewer 2
        Interviewer 3
        Interviewer 1 (Technical Lead)
        Hiring Manager
        HR Manager
        Recruiter
    """

    normalized = normalize_speaker_name(
        speaker_name
    )

    if normalized in INTERVIEWER_LABELS:
        return True

    if re.match(
        r"^interviewer\s*\d+(?:\s+.*)?$",
        normalized,
    ):
        return True

    return False


def is_candidate_label(
    speaker_name: str,
) -> bool:

    normalized = normalize_speaker_name(
        speaker_name
    )

    return normalized in CANDIDATE_LABELS


# ============================================================
# Metadata validation
# ============================================================

def looks_like_person_name(
    text: str,
) -> bool:

    text = text.strip()

    tokens = get_name_tokens(text)

    if not tokens:
        return False

    if tokens[0] in DIALOGUE_STARTERS:
        return False

    if len(tokens) > 4:
        return False

    original_tokens = re.findall(
        r"[A-Za-z][A-Za-z.'-]*",
        text,
    )

    meaningful_tokens = [
        token
        for token in original_tokens
        if normalize_speaker_name(token)
        not in TITLES
    ]

    if not meaningful_tokens:
        return False

    # Person names should generally be capitalized.
    for token in meaningful_tokens:

        if not token[0].isupper():
            return False

    return True


def looks_like_candidate_metadata(
    text: str,
    candidate_name: Optional[str],
) -> bool:

    if not candidate_name:
        return False

    first_part = (
        text
        .split(",", 1)[0]
        .strip()
    )

    return names_match(
        first_part,
        candidate_name,
    )


# ============================================================
# Transcript preprocessing
# ============================================================

def clean_markdown_line(
    line: str,
) -> str:
    """
    Convert:

        **Dr. Kim:** Hello

    to:

        Dr. Kim: Hello
    """

    line = line.strip()

    line = re.sub(
        r"^#+\s*",
        "",
        line,
    )

    line = re.sub(
        r"^\*\*(.+?):\*\*\s*",
        r"\1: ",
        line,
    )

    line = re.sub(
        r"^\*\*",
        "",
        line,
    )

    line = re.sub(
        r"\*\*$",
        "",
        line,
    )

    return line.strip()


def join_split_speaker_lines(
    lines: list[str],
) -> list[str]:
    """
    Repair formatting such as:

        Interviewer: Dr.
        Smith, AI Research Lead

    into:

        Interviewer: Dr. Smith, AI Research Lead


    and:

        Dr.
        Smith: Tell me about your background.

    into:

        Dr. Smith: Tell me about your background.
    """

    repaired = []

    index = 0

    while index < len(lines):

        current = lines[index].strip()

        if index + 1 < len(lines):

            next_line = lines[index + 1].strip()

            # ------------------------------------------------
            # Metadata:
            #
            # Interviewer: Dr.
            # Smith, AI Research Lead
            # ------------------------------------------------

            if re.match(
                r"^Interviewer:\s*(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.$",
                current,
                re.IGNORECASE,
            ):

                repaired.append(
                    current + " " + next_line
                )

                index += 2
                continue


            # ------------------------------------------------
            # Dialogue speaker:
            #
            # Dr.
            # Smith: Hello...
            # ------------------------------------------------

            if (
                re.match(
                    r"^(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.$",
                    current,
                    re.IGNORECASE,
                )
                and
                re.match(
                    r"^[A-Za-z][A-Za-z.'-]*:",
                    next_line,
                )
            ):

                repaired.append(
                    current + " " + next_line
                )

                index += 2
                continue

        repaired.append(current)

        index += 1

    return repaired


def preprocess_transcript(
    transcript: str,
) -> str:

    cleaned_lines = [
        clean_markdown_line(line)
        for line in transcript.splitlines()
    ]

    repaired_lines = join_split_speaker_lines(
        cleaned_lines
    )

    return "\n".join(repaired_lines)


# ============================================================
# Interviewer metadata extraction
# ============================================================

def extract_interviewer_names(
    transcript: str,
) -> list[str]:

    transcript = preprocess_transcript(
        transcript
    )

    interviewer_names = []

    # --------------------------------------------------------
    # Single interviewer metadata
    # --------------------------------------------------------

    for line in transcript.splitlines():

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^Interviewer:\s*(.+)$",
            line,
            re.IGNORECASE,
        )

        if not match:
            continue

        content = (
            match.group(1).strip()
        )

        # Named interviewer metadata generally looks like:
        #
        # Rachel Lee, Senior Engineer
        #
        # rather than normal dialogue.

        if "," not in content:
            continue

        possible_name = (
            content
            .split(",", 1)[0]
            .strip()
        )

        if not looks_like_person_name(
            possible_name
        ):
            continue

        interviewer_names.append(
            possible_name
        )


    # --------------------------------------------------------
    # Explicit panel metadata
    # --------------------------------------------------------

    lines = transcript.splitlines()

    inside_panel = False

    for line in lines:

        stripped = line.strip()

        if stripped.lower() == "interviewees:":
            inside_panel = True
            continue

        if not inside_panel:
            continue

        if (
            stripped.lower()
            .startswith("candidate:")
        ):
            inside_panel = False
            continue

        if not stripped:
            continue

        match = re.match(
            r"^-\s*(.+)$",
            stripped,
        )

        if not match:
            continue

        content = (
            match.group(1).strip()
        )

        possible_name = (
            content
            .split(",", 1)[0]
            .strip()
        )

        if looks_like_person_name(
            possible_name
        ):
            interviewer_names.append(
                possible_name
            )


    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique_names = []

    seen = set()

    for name in interviewer_names:

        normalized = normalize_speaker_name(
            name
        )

        if normalized not in seen:

            seen.add(normalized)

            unique_names.append(
                name
            )

    return unique_names


# ============================================================
# Unknown interviewer inference
# ============================================================

def infer_interviewer_names(
    turns: list[dict],
) -> set[str]:
    """
    Controlled fallback for transcripts such as:

        Alex: Question
        Rahul: Answer
        Alex: Question
        Rahul: Answer

    If an unknown speaker repeatedly occurs immediately
    before the known candidate, infer that speaker as an
    interviewer.

    Requiring at least two transitions avoids treating a
    random unknown speaker as an interviewer based on one
    occurrence.
    """

    counts = {}

    for index in range(
        len(turns) - 1
    ):

        current = turns[index]
        following = turns[index + 1]

        if (
            current["speaker"] == "unknown"
            and
            following["speaker"] == "candidate"
        ):

            name = current["speaker_name"]

            counts[name] = (
                counts.get(name, 0) + 1
            )

    return {
        name
        for name, count in counts.items()
        if count >= 2
    }


# ============================================================
# Transcript parsing
# ============================================================

def parse_transcript(
    transcript: str,
    candidate_name: Optional[str] = None,
) -> list[dict]:

    if not isinstance(transcript, str):
        return []

    if not transcript.strip():
        return []

    transcript = preprocess_transcript(
        transcript
    )

    interviewer_names = extract_interviewer_names(
        transcript
    )

    turns = []

    pattern = re.compile(
        r"^[ \t]*"
        r"([^:\n]+?)"
        r":[ \t]*"
        r"(.*)$",
        re.MULTILINE,
    )

    matches = list(
        pattern.finditer(transcript)
    )


    # ========================================================
    # First pass
    # ========================================================

    for index, match in enumerate(matches):

        raw_speaker_name = (
            match.group(1).strip()
        )

        speaker_name = normalize_speaker_name(
            raw_speaker_name
        )

        first_line = (
            match.group(2).strip()
        )


        # ----------------------------------------------------
        # End of turn
        # ----------------------------------------------------

        if index + 1 < len(matches):

            end_position = (
                matches[index + 1].start()
            )

        else:

            end_position = len(transcript)


        remaining_text = transcript[
            match.end():end_position
        ].strip()


        text = " ".join(
            part.strip()
            for part in [
                first_line,
                remaining_text,
            ]
            if part.strip()
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            continue


        # ----------------------------------------------------
        # Skip obvious metadata fields
        # ----------------------------------------------------

        if speaker_name in METADATA_LABELS:
            continue


        # ====================================================
        # Interviewer labels
        # ====================================================

        if is_interviewer_label(
            raw_speaker_name
        ):

            # Only plain "Interviewer:" can be named
            # interviewer metadata.

            if speaker_name == "interviewer":

                possible_metadata_name = (
                    first_line
                    .split(",", 1)[0]
                    .strip()
                )

                is_metadata = (
                    looks_like_person_name(
                        possible_metadata_name
                    )
                    and
                    any(
                        names_match(
                            possible_metadata_name,
                            interviewer_name,
                        )
                        for interviewer_name
                        in interviewer_names
                    )
                )

                if is_metadata:
                    continue

            speaker = "interviewer"


        # ====================================================
        # Candidate / Interviewee
        # ====================================================

        elif is_candidate_label(
            raw_speaker_name
        ):

            if looks_like_candidate_metadata(
                first_line,
                candidate_name,
            ):
                continue

            speaker = "candidate"


        # ====================================================
        # Named candidate
        # ====================================================

        elif names_match(
            raw_speaker_name,
            candidate_name,
        ):

            speaker = "candidate"


        # ====================================================
        # Named interviewer
        # ====================================================

        elif any(
            names_match(
                raw_speaker_name,
                interviewer_name,
            )
            for interviewer_name
            in interviewer_names
        ):

            speaker = "interviewer"


        # ====================================================
        # Unknown
        # ====================================================

        else:

            speaker = "unknown"


        turns.append(
            {
                "speaker": speaker,
                "speaker_name": speaker_name,
                "text": text,
            }
        )


    # ========================================================
    # Second pass:
    # infer repeated unnamed interviewers
    # ========================================================

    inferred_interviewers = (
        infer_interviewer_names(turns)
    )

    if inferred_interviewers:

        for turn in turns:

            if (
                turn["speaker"] == "unknown"
                and
                turn["speaker_name"]
                in inferred_interviewers
            ):

                turn["speaker"] = "interviewer"


    return turns


# ============================================================
# Q -> A -> Next-Q
# ============================================================

def create_qa_sequences(
    turns: list[dict],
) -> list[dict]:

    sequences = []

    for index in range(
        len(turns) - 1
    ):

        current_turn = turns[index]
        next_turn = turns[index + 1]

        if (
            current_turn["speaker"]
            == "interviewer"
            and
            next_turn["speaker"]
            == "candidate"
        ):

            question = (
                current_turn["text"]
            )

            answer = (
                next_turn["text"]
            )

            next_question = None


            for future_turn in turns[
                index + 2:
            ]:

                if (
                    future_turn["speaker"]
                    == "interviewer"
                ):

                    next_question = (
                        future_turn["text"]
                    )

                    break


            sequences.append(
                {
                    "question": question,
                    "answer": answer,
                    "next_question": next_question,
                }
            )

    return sequences