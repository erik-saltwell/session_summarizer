# Backchannel Identification Architecture Decisions

- You do **not** want to train your own model to identify backchannels.

- You are willing to use **existing models** where helpful, but the core system should still work with a **rule-based / role-based architecture**.

- The recommended architecture is a **post-diarization identification layer**, not a standalone trained backchannel classifier.

- The scope is **only identification**, not transcript rendering, analytics views, or other downstream handling decisions.

- You have a key hard constraint: **only one mono audio channel**.

- Because of that mono constraint, you do **not** want an architecture that depends on reliably modeling rich multi-party roles like **floor-holder / listener / challenger**.

- For your purposes, the useful conversational roles are reduced to:
  - **Speaker**
  - **Challenger**

- The recommended existing-model family for ambiguous cases is **VAP (Voice Activity Projection)**.

- The specific VAP system recommended is **Erik Ekstedt’s VAP / VoiceActivityProjection**.

- Because your audio is **mixed mono conversation audio**, you should **not** feed the raw mono waveform directly into VAP as though it were natively suitable for backchannel detection.

- If you use VAP, the recommended integration is to create **synthetic pairwise stereo windows** from:
  - the mono waveform
  - diarized segment timing
  - local speaker masks

- In those synthetic windows:
  - one channel represents the **current active speaker**
  - the other channel represents the **candidate challenger**

- For conversations with more than two people, VAP should still be applied **pairwise**, one candidate challenger at a time against the locally dominant speaker.

- VAP should be used as an **ambiguity resolver**, not as the primary detector.

- The first-pass detector should be **rules-based**, driven by:
  - diarization
  - overlap/timing
  - ASR text
  - question detection

- Short non-dominant segments should not be treated as a simple binary. The recommended identification categories are:
  - **BACKCHANNEL**
  - **SHORT_ANSWER**
  - **FLOOR_TAKE_ATTEMPT**
  - **SPEECH / OTHER SHORT EVENT**

- The practical rule priority is:
  - first decide whether a segment is even a **short-event candidate**
  - then prefer **SHORT_ANSWER** if there is strong question-response evidence
  - then prefer **FLOOR_TAKE_ATTEMPT** if there is strong turn-grab evidence
  - otherwise, if the dominant speaker continues and the token/timing fit, classify as **BACKCHANNEL**
  - if none of those are reliable, leave it as **OTHER SHORT EVENT / SPEECH**

- The current candidate-generation rules for a possible backchannel are:
  - segment is short, roughly `<= 1.0–1.2 s`
  - segment belongs to someone other than the currently dominant speaker
  - segment overlaps the dominant speaker or occurs very near the dominant speaker’s continuation point
  - there is evidence the dominant speaker continues after it

- The current rules for classifying **BACKCHANNEL** are:
  - short duration
  - produced by the non-dominant speaker
  - overlaps the other speaker, or starts right near the other speaker’s continuation boundary
  - the original speaker keeps talking or quickly resumes
  - lexical content is a known backchannel token or vocal acknowledgment
  - there is **not** strong evidence that the segment is answering a question or taking the floor

- The current lexical cues for **BACKCHANNEL** are grouped like this:
  - very strong:
    - `mhm`
    - `mm-hm`
    - `uh-huh`
    - 'sure'
  - medium:
    - `right`
    - `okay`
    - `oh`
  - ambiguous:
    - `yeah`
    - `yes`
    - `no`

- The current rules for classifying **SHORT_ANSWER** are:
  - the preceding dominant-speaker segment is likely a question
  - the short utterance is semantically responsive
  - after the short utterance, the respondent continues or becomes the next dominant speaker
  - ambiguous tokens like `yeah`, `yes`, `no`, `right`, `okay` often belong here when they resolve a question

- The current rules for classifying **FLOOR_TAKE_ATTEMPT** are:
  - the segment starts in overlap or at a transition point
  - the segment is short but substantial enough to launch a turn
  - after it, the original speaker stops or yields
  - the short-segment speaker continues into a longer turn

- The current rules for classifying **SPEECH / OTHER SHORT EVENT** are:
  - the segment is short but does not clearly match backchannel, short answer, or floor-take behavior
  - the evidence is mixed or weak
  - lexical content is not informative
  - diarization/overlap evidence is uncertain

- Because your transcripts do **not** have punctuation, identifying whether the previous segment is a question should be done with a **stacked question detector**, not a punctuation-based rule.

- The recommended question detector uses three sources of evidence:
  - **lexical form**
  - **local dialogue context**
  - **prosody from the audio**

- The recommended **lexical question cues** are:
  - strong yes/no openers:
    - `do`, `does`, `did`, `are`, `is`, `was`, `were`, `can`, `could`, `will`, `would`, `have`, `has`, `had`, `should`
  - strong wh-words:
    - `what`, `when`, `where`, `why`, `who`, `whom`, `whose`, `which`, `how`
  - question-tag or trailing confirmation forms:
    - `right`, `okay`, `yeah`, `or no`, and similar confirmation endings

- The recommended **contextual question cues** are:
  - the next speaker answers directly
  - the current speaker yields the floor
  - the next segment contains a direct response such as `yes`, `no`, `yeah`, `right`, a value, a time, or a choice

- The recommended **prosodic question cues** are:
  - rising or late-rising final pitch
  - final pitch movement that is high relative to the speaker’s recent baseline
  - final lengthening
  - a pause after the utterance that invites response
  - reduced continuation by the same speaker

- The recommended question detector should produce a **question score** and map it roughly to:
  - `LIKELY_QUESTION`
  - `POSSIBLE_QUESTION`
  - `NOT_QUESTION`

- The recommended weighting scheme for question scoring is:
  - wh-word at start: `+3`
  - auxiliary/modal inversion at start: `+3`
  - explicit choice phrase like `or`: `+1`
  - final rising pitch: `+2`
  - final lengthening: `+1`
  - next turn answers directly: `+2`
  - same speaker continues seamlessly: `-2`

- The recommended implementation order for question detection is:
  - **first pass:** text-only question guess
  - **second pass:** prosody correction for ambiguous segments
  - **third pass:** conversational correction using the next turn

- For backchannel identification, the question detector is used as a **veto / rerouting signal**:
  - if the previous segment is `LIKELY_QUESTION`, be much less willing to label the next short response as `BACKCHANNEL`
  - prefer `SHORT_ANSWER` instead

- The recommended reusable packages/models for question detection are:
  - **Praat-Parselmouth** as the preferred prosody package
  - **deepmultilingualpunctuation** as a cheap first-pass punctuation/question guesser
  - **Hugging Face zero-shot classification** as an optional ambiguity resolver for text cases

- The recommendation is **not** to rely on punctuation restoration alone for question detection.

- The final identification strategy currently stored in the conversation is:
  - **use diarization first**
  - **use rules to nominate and classify most short events**
  - **use stacked question detection to distinguish backchannels from short answers**
  - **use pairwise synthetic-stereo VAP only for ambiguous mono cases**
  - **make the final identification decision in your own rule layer**
