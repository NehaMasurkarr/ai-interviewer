from enum import Enum
from typing import Dict, List


class CompetencyState(Enum):
    """
    Represents how thoroughly a competency has been
    covered during the interview.
    """

    NOT_COVERED = "NOT_COVERED"
    MENTIONED = "MENTIONED"
    EXPLORED = "EXPLORED"
    ASSESSED = "ASSESSED"


STATE_RANK = {
    CompetencyState.NOT_COVERED: 0,
    CompetencyState.MENTIONED: 1,
    CompetencyState.EXPLORED: 2,
    CompetencyState.ASSESSED: 3,
}


class CompetencyTracker:
    """
    Maintains interview coverage state for each competency.

    Competency states are monotonic:

    NOT_COVERED
        ->
    MENTIONED
        ->
    EXPLORED
        ->
    ASSESSED

    A competency can never move backward.
    """

    def __init__(
        self,
        competencies: List[str],
    ):
        self._states: Dict[
            str,
            CompetencyState,
        ] = {
            competency: CompetencyState.NOT_COVERED
            for competency in competencies
        }


    def get_state(
        self,
        competency: str,
    ) -> CompetencyState:
        """
        Return the current state of a competency.
        """

        if competency not in self._states:
            raise KeyError(
                f"Unknown competency: {competency}"
            )

        return self._states[competency]


    def get_states(
        self,
    ) -> Dict[str, CompetencyState]:
        """
        Return a copy of all competency states.
        """

        return dict(
            self._states
        )


    def update(
        self,
        competency: str,
        new_state: CompetencyState,
    ) -> bool:
        """
        Update a competency only if the new state
        represents greater interview coverage.

        Returns True when the state changed.
        Returns False when no change was made.
        """

        if competency not in self._states:
            raise KeyError(
                f"Unknown competency: {competency}"
            )

        current_state = self._states[
            competency
        ]

        current_rank = STATE_RANK[
            current_state
        ]

        new_rank = STATE_RANK[
            new_state
        ]

        if new_rank <= current_rank:
            return False

        self._states[
            competency
        ] = new_state

        return True


    def apply_updates(
        self,
        updates: Dict[
            str,
            CompetencyState,
        ],
    ) -> Dict[
        str,
        CompetencyState,
    ]:
        """
        Apply multiple competency updates.

        Unknown competencies are ignored because an LLM
        response should never be allowed to introduce new
        interview targets.

        Returns only the states that actually changed.
        """

        changed = {}

        for competency, new_state in (
            updates.items()
        ):

            if competency not in self._states:
                continue

            was_updated = self.update(
                competency=competency,
                new_state=new_state,
            )

            if was_updated:
                changed[
                    competency
                ] = new_state

        return changed


    def get_unassessed(
        self,
    ) -> List[str]:
        """
        Return competencies that have not reached ASSESSED.
        """

        return [
            competency
            for competency, state
            in self._states.items()
            if state != CompetencyState.ASSESSED
        ]


    def get_assessed(
        self,
    ) -> List[str]:
        """
        Return competencies that reached ASSESSED.
        """

        return [
            competency
            for competency, state
            in self._states.items()
            if state == CompetencyState.ASSESSED
        ]


    def all_assessed(
        self,
    ) -> bool:
        """
        Return True when every competency is ASSESSED.
        """

        if not self._states:
            return True

        return all(
            state == CompetencyState.ASSESSED
            for state in self._states.values()
        )


    def format_status(
        self,
    ) -> str:
        """
        Return human-readable competency state.
        """

        return "\n".join(
            (
                f"- {competency}: "
                f"{state.value}"
            )
            for competency, state
            in self._states.items()
        )


def main():
    """
    Test monotonic competency updates.
    """

    tracker = CompetencyTracker(
        competencies=[
            "Machine Learning",
            "SQL",
            "Communication",
        ]
    )

    print("=" * 80)
    print("COMPETENCY TRACKER TEST")
    print("=" * 80)

    print("\nINITIAL STATE")
    print(
        tracker.format_status()
    )

    print("\nUPDATING SQL -> MENTIONED")

    tracker.update(
        "SQL",
        CompetencyState.MENTIONED,
    )

    print(
        tracker.format_status()
    )

    print("\nUPDATING SQL -> EXPLORED")

    tracker.update(
        "SQL",
        CompetencyState.EXPLORED,
    )

    print(
        tracker.format_status()
    )

    print("\nUPDATING SQL -> ASSESSED")

    tracker.update(
        "SQL",
        CompetencyState.ASSESSED,
    )

    print(
        tracker.format_status()
    )

    print(
        "\nATTEMPTING SQL -> MENTIONED"
    )

    changed = tracker.update(
        "SQL",
        CompetencyState.MENTIONED,
    )

    print(
        f"State changed: {changed}"
    )

    print(
        f"SQL remains: "
        f"{tracker.get_state('SQL').value}"
    )

    print("\nBATCH UPDATE")

    changed_states = (
        tracker.apply_updates(
            {
                "Machine Learning":
                    CompetencyState.EXPLORED,

                "SQL":
                    CompetencyState.NOT_COVERED,

                "Communication":
                    CompetencyState.MENTIONED,

                "Fake Competency":
                    CompetencyState.ASSESSED,
            }
        )
    )

    print("\nCHANGED")

    for competency, state in (
        changed_states.items()
    ):
        print(
            f"- {competency}: "
            f"{state.value}"
        )

    print("\nFINAL STATE")

    print(
        tracker.format_status()
    )

    print("\nASSESSED")

    print(
        tracker.get_assessed()
    )

    print("\nUNASSESSED")

    print(
        tracker.get_unassessed()
    )


if __name__ == "__main__":
    main()