import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.agent.interviewer_agent import QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.policy.interview_policy import InterviewPhase, InterviewPolicyConfig
from src.policy.question_source import QuestionSource
from src.profile.candidate_profile import (
    CandidateProfile,
    Experience,
    Project,
    ResumeClaim,
)
from src.state.competency_tracker import CompetencyState


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


class InterviewSessionError(ValueError):
    """Raised when portable interview-session data is invalid."""


@dataclass(frozen=True)
class InterviewSession:
    """Versioned, JSON-compatible interview session schema."""

    schema_version: int
    candidate_profile: Dict[str, Any]
    job_profile: Dict[str, Any]
    job_description: str
    interview_plan: Dict[str, Any]
    coordinator: Dict[str, Any]
    engine: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_profile": self.candidate_profile,
            "job_profile": self.job_profile,
            "job_description": self.job_description,
            "interview_plan": self.interview_plan,
            "coordinator": self.coordinator,
            "engine": self.engine,
        }


@dataclass(frozen=True)
class _DecodedSession:
    candidate_profile: CandidateProfile
    job_profile: JobProfile
    job_description: str
    interview_plan: InterviewPlan
    max_decision_attempts: int
    current_question: str
    current_question_type: Optional[QuestionType]
    current_target_competency: Optional[str]
    current_question_source: QuestionSource
    phase: InterviewPhase
    competency_states: Dict[str, CompetencyState]
    history: List[Dict[str, str]]
    policy_config: InterviewPolicyConfig
    followup_counts: Dict[str, int]
    behavioral_questions_completed: int
    question_source_counts: Dict[QuestionSource, int]


def serialize_interview_session(coordinator) -> Dict[str, Any]:
    """Serialize an InterviewCoordinator to plain session data."""

    engine = coordinator.engine
    policy = engine.policy
    session = InterviewSession(
        schema_version=SCHEMA_VERSION,
        candidate_profile=_candidate_to_dict(coordinator.candidate_profile),
        job_profile=_job_to_dict(coordinator.job_profile),
        job_description=coordinator.job_description,
        interview_plan=_plan_to_dict(coordinator.interview_plan),
        coordinator={
            "max_decision_attempts": coordinator.resolver.max_attempts,
        },
        engine={
            "role": engine.role,
            "current_question": engine.current_question,
            "current_question_type": (
                engine.current_question_type.value
                if engine.current_question_type is not None
                else None
            ),
            "current_target_competency": (
                engine.current_target_competency
            ),
            "current_question_source": engine.current_question_source.value,
            "phase": policy.phase.value,
            "competency_states": {
                competency: state.value
                for competency, state
                in engine.get_competency_states().items()
            },
            "history": engine.memory.get_history(),
            "policy": {
                "config": {
                    "max_followups_per_target": (
                        policy.config.max_followups_per_target
                    ),
                    "behavioral_questions_required": (
                        policy.config.behavioral_questions_required
                    ),
                    "require_jd_technical": policy.config.require_jd_technical,
                    "require_jd_scenario": policy.config.require_jd_scenario,
                    "require_resume_validation_when_evidence": (
                        policy.config.require_resume_validation_when_evidence
                    ),
                },
                "followup_counts": dict(policy.followup_counts),
                "behavioral_questions_completed": (
                    policy.behavioral_questions_completed
                ),
                "question_source_counts": {
                    source.value: count
                    for source, count in policy.question_source_counts.items()
                },
            },
        },
    )
    data = session.to_dict()

    # Apply the same validation used by restoration so corrupt
    # in-memory state is never emitted as a valid session.
    _decode_session(data)
    return data


def restore_interview_session(
    session_data: Dict[str, Any],
    decision_generator,
    *,
    coordinator_factory: Optional[Callable[..., Any]] = None,
):
    """Rebuild an InterviewCoordinator from validated session data."""

    decoded = _decode_session(session_data)

    if not callable(decision_generator):
        raise TypeError("decision_generator must be callable.")

    if coordinator_factory is None:
        from src.agent.interview_coordinator import InterviewCoordinator

        coordinator_factory = InterviewCoordinator

    coordinator = coordinator_factory(
        candidate_profile=decoded.candidate_profile,
        job_profile=decoded.job_profile,
        job_description=decoded.job_description,
        decision_generator=decision_generator,
        interview_plan=decoded.interview_plan,
        policy_config=decoded.policy_config,
        max_decision_attempts=decoded.max_decision_attempts,
    )
    engine = coordinator.engine

    engine.competency_tracker.apply_updates(decoded.competency_states)
    engine._sync_plan_states()

    for turn in decoded.history:
        engine.memory.add_turn(turn["question"], turn["answer"])

    for competency, count in decoded.followup_counts.items():
        for _ in range(count):
            if not engine.record_followup(competency):
                raise InterviewSessionError(
                    "Failed to restore validated follow-up state."
                )

    for _ in range(decoded.behavioral_questions_completed):
        if not engine.record_behavioral_question():
            raise InterviewSessionError(
                "Failed to restore validated behavioral state."
            )

    engine.policy.question_source_counts = {
        source: 0 for source in QuestionSource
    }
    for source, count in decoded.question_source_counts.items():
        for _ in range(count):
            engine.policy.record_question_source(source)

    engine.transition_to_phase(decoded.phase)
    engine.current_question = decoded.current_question
    engine.current_question_type = decoded.current_question_type
    engine.current_target_competency = decoded.current_target_competency
    engine.current_question_source = decoded.current_question_source

    return coordinator


def session_to_json(coordinator, **json_options) -> str:
    """Serialize a coordinator directly to JSON text."""

    options = {"sort_keys": True}
    options.update(json_options)
    return json.dumps(serialize_interview_session(coordinator), **options)


def restore_interview_session_json(
    json_text: str,
    decision_generator,
    **restore_options,
):
    """Restore a coordinator from JSON text."""

    try:
        data = json.loads(json_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise InterviewSessionError("Session JSON is invalid.") from error

    return restore_interview_session(
        data,
        decision_generator,
        **restore_options,
    )


def _decode_session(data: Dict[str, Any]) -> _DecodedSession:
    root = _require_dict(data, "session")
    version = _require_int(root.get("schema_version"), "schema_version")

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise InterviewSessionError(
            f"Unsupported interview session schema version: {version}."
        )

    candidate = _candidate_from_dict(
        _required(root, "candidate_profile", "session")
    )
    job = _job_from_dict(_required(root, "job_profile", "session"))
    job_description = _require_nonempty_string(
        _required(root, "job_description", "session"),
        "job_description",
    )
    plan = _plan_from_dict(_required(root, "interview_plan", "session"))
    coordinator = _require_dict(
        _required(root, "coordinator", "session"), "coordinator"
    )
    max_attempts = _require_int(
        _required(coordinator, "max_decision_attempts", "coordinator"),
        "coordinator.max_decision_attempts",
        minimum=1,
    )
    engine = _require_dict(_required(root, "engine", "session"), "engine")
    role = _require_nonempty_string(
        _required(engine, "role", "engine"), "engine.role"
    )

    if role != plan.role:
        raise InterviewSessionError(
            "Engine role does not match the interview plan role."
        )

    current_question = _require_nonempty_string(
        _required(engine, "current_question", "engine"),
        "engine.current_question",
    )
    question_type = _optional_enum(
        engine.get("current_question_type"),
        QuestionType,
        "engine.current_question_type",
    )
    target = engine.get("current_target_competency")
    if target is not None:
        target = _require_nonempty_string(
            target, "engine.current_target_competency"
        )
    phase = _required_enum(engine, "phase", InterviewPhase, "engine")
    if version == 1:
        current_source = _legacy_question_source(question_type, phase)
    else:
        current_source = _required_enum(
            engine, "current_question_source", QuestionSource, "engine"
        )

    target_names = [target.competency for target in plan.targets]
    if len(target_names) != len(set(target_names)):
        raise InterviewSessionError(
            "Interview plan contains duplicate competencies."
        )

    states_data = _require_dict(
        _required(engine, "competency_states", "engine"),
        "engine.competency_states",
    )
    if set(states_data) != set(target_names):
        raise InterviewSessionError(
            "Plan and competency-state targets do not match."
        )
    states = {
        name: _enum_value(
            states_data[name],
            CompetencyState,
            f"engine.competency_states.{name}",
        )
        for name in target_names
    }

    for plan_target in plan.targets:
        if plan_target.state != states[plan_target.competency]:
            raise InterviewSessionError(
                "Interview plan state does not match competency state "
                f"for {plan_target.competency}."
            )

    _validate_question_metadata(
        question_type, target, current_source, phase, set(target_names)
    )
    history = _history_from_data(_required(engine, "history", "engine"))
    policy_data = _require_dict(
        _required(engine, "policy", "engine"), "engine.policy"
    )
    config_data = _require_dict(
        _required(policy_data, "config", "engine.policy"),
        "engine.policy.config",
    )
    max_followups = _require_int(
        _required(
            config_data,
            "max_followups_per_target",
            "engine.policy.config",
        ),
        "engine.policy.config.max_followups_per_target",
        minimum=0,
    )
    behavioral_required = _require_int(
        _required(
            config_data,
            "behavioral_questions_required",
            "engine.policy.config",
        ),
        "engine.policy.config.behavioral_questions_required",
        minimum=0,
    )
    require_jd_technical = _optional_bool(
        config_data, "require_jd_technical", default=True
    )
    require_jd_scenario = _optional_bool(
        config_data, "require_jd_scenario", default=True
    )
    require_resume = _optional_bool(
        config_data, "require_resume_validation_when_evidence", default=True
    )
    followups_data = _require_dict(
        _required(policy_data, "followup_counts", "engine.policy"),
        "engine.policy.followup_counts",
    )
    if not set(followups_data).issubset(set(target_names)):
        raise InterviewSessionError(
            "Follow-up state references an unknown competency."
        )
    followups = {
        name: _require_int(
            count,
            f"engine.policy.followup_counts.{name}",
            minimum=0,
            maximum=max_followups,
        )
        for name, count in followups_data.items()
    }
    behavioral_completed = _require_int(
        _required(
            policy_data,
            "behavioral_questions_completed",
            "engine.policy",
        ),
        "engine.policy.behavioral_questions_completed",
        minimum=0,
        maximum=behavioral_required,
    )
    if version == 1:
        source_counts = {source: 0 for source in QuestionSource}
        source_counts[QuestionSource.OPENING] = 1
        if question_type is not None:
            source_counts[current_source] = 1
    else:
        raw_source_counts = _require_dict(
            _required(policy_data, "question_source_counts", "engine.policy"),
            "engine.policy.question_source_counts",
        )
        if set(raw_source_counts) != {source.value for source in QuestionSource}:
            raise InterviewSessionError(
                "Question-source counts must contain every known source."
            )
        source_counts = {
            source: _require_int(
                raw_source_counts[source.value],
                f"engine.policy.question_source_counts.{source.value}",
                minimum=0,
            )
            for source in QuestionSource
        }
        if (
            question_type is not None
            and source_counts[current_source] < 1
        ):
            raise InterviewSessionError(
                "Current question source has no accepted-question count."
            )

    return _DecodedSession(
        candidate_profile=candidate,
        job_profile=job,
        job_description=job_description,
        interview_plan=plan,
        max_decision_attempts=max_attempts,
        current_question=current_question,
        current_question_type=question_type,
        current_target_competency=target,
        current_question_source=current_source,
        phase=phase,
        competency_states=states,
        history=history,
        policy_config=InterviewPolicyConfig(
            max_followups_per_target=max_followups,
            behavioral_questions_required=behavioral_required,
            require_jd_technical=require_jd_technical,
            require_jd_scenario=require_jd_scenario,
            require_resume_validation_when_evidence=require_resume,
        ),
        followup_counts=followups,
        behavioral_questions_completed=behavioral_completed,
        question_source_counts=source_counts,
    )


def _validate_question_metadata(question_type, target, source, phase, targets):
    if question_type is None:
        if (
            target is not None
            or phase != InterviewPhase.INTRODUCTION
            or source != QuestionSource.OPENING
        ):
            raise InterviewSessionError(
                "Opening question metadata is inconsistent with phase."
            )
        return

    if question_type == QuestionType.CLOSING:
        if target is not None or source != QuestionSource.CLOSING or phase not in {
            InterviewPhase.CLOSING,
            InterviewPhase.COMPLETE,
        }:
            raise InterviewSessionError("Closing question metadata is invalid.")
        return

    if target not in targets:
        raise InterviewSessionError(
            "Current question references an unknown competency."
        )

    expected_phase = {
        QuestionType.NEW_TARGET: InterviewPhase.TECHNICAL,
        QuestionType.FOLLOW_UP: InterviewPhase.TECHNICAL,
        QuestionType.BEHAVIORAL: InterviewPhase.BEHAVIORAL,
    }[question_type]
    if phase != expected_phase:
        raise InterviewSessionError(
            "Current question type is inconsistent with interview phase."
        )
    if question_type == QuestionType.BEHAVIORAL:
        if source != QuestionSource.BEHAVIORAL:
            raise InterviewSessionError("Behavioral question source is invalid.")
    elif source not in {
        QuestionSource.RESUME_VALIDATION,
        QuestionSource.JD_TECHNICAL,
        QuestionSource.JD_SCENARIO,
    }:
        raise InterviewSessionError("Technical question source is invalid.")


def _legacy_question_source(question_type, phase):
    if question_type is None:
        return QuestionSource.OPENING
    if question_type == QuestionType.BEHAVIORAL:
        return QuestionSource.BEHAVIORAL
    if question_type == QuestionType.CLOSING:
        return QuestionSource.CLOSING
    return QuestionSource.JD_TECHNICAL


def _candidate_to_dict(profile: CandidateProfile) -> Dict[str, Any]:
    return {
        "name": profile.name,
        "education": list(profile.education),
        "experiences": [
            {
                "title": item.title,
                "company": item.company,
                "description": list(item.description),
            }
            for item in profile.experiences
        ],
        "projects": [
            {"name": item.name, "description": list(item.description)}
            for item in profile.projects
        ],
        "skills": list(profile.skills),
        "certifications": list(profile.certifications),
        "claims": [
            {
                "claim": item.claim,
                "source": item.source,
                "technologies": list(item.technologies),
            }
            for item in profile.claims
        ],
    }


def _candidate_from_dict(data: Any) -> CandidateProfile:
    value = _require_dict(data, "candidate_profile")
    return CandidateProfile(
        name=_require_string(_required(value, "name", "candidate_profile"), "candidate_profile.name"),
        education=_string_list(_required(value, "education", "candidate_profile"), "candidate_profile.education"),
        experiences=[
            Experience(
                title=_require_string(_required(item, "title", path), f"{path}.title"),
                company=_require_string(_required(item, "company", path), f"{path}.company"),
                description=_string_list(_required(item, "description", path), f"{path}.description"),
            )
            for item, path in _dict_items(_required(value, "experiences", "candidate_profile"), "candidate_profile.experiences")
        ],
        projects=[
            Project(
                name=_require_string(_required(item, "name", path), f"{path}.name"),
                description=_string_list(_required(item, "description", path), f"{path}.description"),
            )
            for item, path in _dict_items(_required(value, "projects", "candidate_profile"), "candidate_profile.projects")
        ],
        skills=_string_list(_required(value, "skills", "candidate_profile"), "candidate_profile.skills"),
        certifications=_string_list(_required(value, "certifications", "candidate_profile"), "candidate_profile.certifications"),
        claims=[
            ResumeClaim(
                claim=_require_string(_required(item, "claim", path), f"{path}.claim"),
                source=_require_string(_required(item, "source", path), f"{path}.source"),
                technologies=_string_list(_required(item, "technologies", path), f"{path}.technologies"),
            )
            for item, path in _dict_items(_required(value, "claims", "candidate_profile"), "candidate_profile.claims")
        ],
    )


def _job_to_dict(profile: JobProfile) -> Dict[str, Any]:
    return {
        "role": profile.role,
        "company": profile.company,
        "summary": profile.summary,
        "requirements": [
            {
                "name": item.name,
                "priority": item.priority,
                "evidence_expected": list(item.evidence_expected),
            }
            for item in profile.requirements
        ],
        "responsibilities": list(profile.responsibilities),
        "preferred_qualifications": list(profile.preferred_qualifications),
    }


def _job_from_dict(data: Any) -> JobProfile:
    value = _require_dict(data, "job_profile")
    requirements = []
    for item, path in _dict_items(
        _required(value, "requirements", "job_profile"),
        "job_profile.requirements",
    ):
        requirements.append(
            JobRequirement(
                name=_require_nonempty_string(_required(item, "name", path), f"{path}.name"),
                priority=_require_nonempty_string(_required(item, "priority", path), f"{path}.priority"),
                evidence_expected=_string_list(_required(item, "evidence_expected", path), f"{path}.evidence_expected"),
            )
        )
    if not requirements:
        raise InterviewSessionError("job_profile.requirements cannot be empty.")
    return JobProfile(
        role=_require_nonempty_string(_required(value, "role", "job_profile"), "job_profile.role"),
        company=_require_string(_required(value, "company", "job_profile"), "job_profile.company"),
        summary=_require_string(_required(value, "summary", "job_profile"), "job_profile.summary"),
        requirements=requirements,
        responsibilities=_string_list(_required(value, "responsibilities", "job_profile"), "job_profile.responsibilities"),
        preferred_qualifications=_string_list(_required(value, "preferred_qualifications", "job_profile"), "job_profile.preferred_qualifications"),
    )


def _plan_to_dict(plan: InterviewPlan) -> Dict[str, Any]:
    return {
        "role": plan.role,
        "targets": [
            {
                "competency": item.competency,
                "priority": item.priority,
                "reason": item.reason,
                "resume_evidence": list(item.resume_evidence),
                "evidence_expected": list(item.evidence_expected),
                "state": item.state.value,
            }
            for item in plan.targets
        ],
    }


def _plan_from_dict(data: Any) -> InterviewPlan:
    value = _require_dict(data, "interview_plan")
    targets = []
    for item, path in _dict_items(
        _required(value, "targets", "interview_plan"),
        "interview_plan.targets",
    ):
        targets.append(
            InterviewTarget(
                competency=_require_nonempty_string(_required(item, "competency", path), f"{path}.competency"),
                priority=_require_nonempty_string(_required(item, "priority", path), f"{path}.priority"),
                reason=_require_string(_required(item, "reason", path), f"{path}.reason"),
                resume_evidence=_string_list(_required(item, "resume_evidence", path), f"{path}.resume_evidence"),
                evidence_expected=_string_list(_required(item, "evidence_expected", path), f"{path}.evidence_expected"),
                state=_required_enum(item, "state", CompetencyState, path),
            )
        )
    if not targets:
        raise InterviewSessionError("interview_plan.targets cannot be empty.")
    return InterviewPlan(
        role=_require_nonempty_string(_required(value, "role", "interview_plan"), "interview_plan.role"),
        targets=targets,
    )


def _history_from_data(data: Any) -> List[Dict[str, str]]:
    history = []
    for item, path in _dict_items(data, "engine.history"):
        history.append(
            {
                "question": _require_nonempty_string(_required(item, "question", path), f"{path}.question"),
                "answer": _require_nonempty_string(_required(item, "answer", path), f"{path}.answer"),
            }
        )
    return history


def _required(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise InterviewSessionError(f"Missing required field: {path}.{key}.")
    return data[key]


def _require_dict(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise InterviewSessionError(f"{path} must be an object.")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise InterviewSessionError(f"{path} must be a string.")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    value = _require_string(value, path)
    if not value.strip():
        raise InterviewSessionError(f"{path} cannot be empty.")
    return value


def _require_int(value: Any, path: str, minimum=None, maximum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InterviewSessionError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise InterviewSessionError(f"{path} cannot be less than {minimum}.")
    if maximum is not None and value > maximum:
        raise InterviewSessionError(f"{path} cannot exceed {maximum}.")
    return value


def _optional_bool(data: Dict[str, Any], key: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise InterviewSessionError(f"engine.policy.config.{key} must be boolean.")
    return value


def _string_list(value: Any, path: str) -> List[str]:
    if not isinstance(value, list):
        raise InterviewSessionError(f"{path} must be a list.")
    return [_require_string(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _dict_items(value: Any, path: str):
    if not isinstance(value, list):
        raise InterviewSessionError(f"{path} must be a list.")
    return [
        (_require_dict(item, f"{path}[{index}]"), f"{path}[{index}]")
        for index, item in enumerate(value)
    ]


def _enum_value(value: Any, enum_type, path: str):
    if not isinstance(value, str):
        raise InterviewSessionError(f"{path} must be a string enum value.")
    try:
        return enum_type(value)
    except ValueError as error:
        raise InterviewSessionError(f"Invalid {path}: {value}.") from error


def _required_enum(data, key, enum_type, path):
    return _enum_value(_required(data, key, path), enum_type, f"{path}.{key}")


def _optional_enum(value, enum_type, path):
    if value is None:
        return None
    return _enum_value(value, enum_type, path)
