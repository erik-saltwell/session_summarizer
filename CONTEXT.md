# Session Summarizer

Session Summarizer processes diarized tabletop roleplaying game transcripts into cleaner, speaker-aware session artifacts.

## Language

**Input Speaker Label**:
An anonymous diarization label assigned to spoken transcript material before participant identity is known.
_Avoid_: Player, character, role

**Speaker Identity**:
The display name assigned to transcript material after the system has identified who or what role the speaker represents.
_Avoid_: Diarization label

**Inferred Role**:
The game-facing role inferred for an input speaker label, either **Game Master** or the name of the character controlled by that participant.
_Avoid_: Player name

**Participant Role Map**:
A mapping from real-world participant name to their inferred game-facing role for a session.
_Avoid_: Character-only map

**Unknown Speaker**:
The fallback speaker identity used when an input speaker label cannot be mapped to an inferred role.
_Avoid_: Unassigned speaker, anonymous

**Session Override Settings**:
The per-session settings file that captures inferred or manually corrected facts for one recorded session.
_Avoid_: Base settings

## Relationships

- An **Input Speaker Label** may map to exactly one **Inferred Role**
- A **Speaker Identity** may be set from an **Inferred Role**
- Multiple **Input Speaker Labels** may map to the same **Inferred Role** when the transcript strongly suggests a diarization split
- A clip with multiple **Input Speaker Labels** receives a **Speaker Identity** only when every mapped label resolves to the same **Inferred Role**
- An unmapped **Input Speaker Label** receives the **Unknown Speaker** identity
- **Session Override Settings** may preserve inferred attendees and player-to-role mappings for exactly one session
- A **Participant Role Map** includes the Game Master as a participant whose role is **Game Master**

## Example Dialogue

> **Dev:** "Should label `2` become the player's real name or their character name?"
> **Domain expert:** "Use the **Inferred Role** as the **Speaker Identity** so the transcript reads by character role, while preserving `2` as the **Input Speaker Label**."

## Flagged Ambiguities

- "label" can mean either **Input Speaker Label** or **Speaker Identity**; resolved: raw diarization labels remain unchanged, and inferred roles are written as speaker identities.
- Multi-speaker clips can contain overlapping **Input Speaker Labels**; resolved: preserve the existing identity when mapped labels imply different inferred roles.
