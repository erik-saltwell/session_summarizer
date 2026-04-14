# User Message Template

Below is the recommended structure for the user message you send alongside the system prompt.

---

```
<session>
  <campaign>Gods Teeth</campaign>
  <session_name>The Cornucopia Job</session_name>
  <date>2026-04-10</date>
</session>

<campaign_context>
  <players>
    - Erik plays Agent Jonas Friday
    - Sarah plays Agent Diana Cross
    - Mike plays Agent Carl Renko
  </players>

  <glossary>
    - Cornucopia Foundation: A front organization with ties to the CDC
    - The Program: Delta Green's officially sanctioned arm
    - OPERA: Codename for the current operation
    - Agent Morrison: Handler, reports to someone unknown outside the Program
  </glossary>
</campaign_context>

<transcript>
[Full diarized transcript here]
</transcript>
```

---

## Design Rationale

**Why XML tags for the top-level sections:**
Anthropic's docs recommend XML tags as the primary structural separator for Claude prompts. They create unambiguous boundaries between metadata, context, and content. Claude was specifically trained to recognize XML tags as organizational markers.

**Why simple dashed lists inside the tags (not YAML, not nested XML):**
- The player and glossary data is simple enough that flat lists are the most readable and maintainable format for you as the author.
- YAML would work but adds ceremony (colons, indentation rules) for no real parsing benefit — Claude handles both equally well.
- Deeply nested XML for each player/glossary entry would bloat the message without improving Claude's comprehension.

**Why the transcript goes last:**
Anthropic's long-context guidance says to put longform data at the top. However, in this case the system prompt already contains all the instructions, so the user message is purely data. Within the user message, putting metadata and context *before* the transcript means Claude reads the player names and glossary first and can apply them while processing the transcript. This is the more natural reading order.

**Why `<campaign_context>` wraps players and glossary together:**
These are both reference material that persists across sessions. Grouping them signals to Claude that this is standing context, not session-specific content. This also means you can maintain a single campaign context block and reuse it across sessions — only swapping out the `<session>` and `<transcript>` blocks each time.

## Notes on Maintenance

- **The glossary is cumulative.** Add new proper nouns as they appear in your campaign. You don't need to be exhaustive — only include terms where Claude might otherwise be confused or lose important context.
- **The player roster rarely changes.** Update it only when players join, leave, or change characters.
- **The session block is the only thing you rewrite each time.**

## Transcript format
The transcript consists of a series of statements, where each statement is as follows:
- The name of the player, in bold, followed by a colon and then a newline
- The text of the statement or clip, followed by a newline
- An additional newline to seperate this statement from the next.

### example transcript:
**erik**:
The person in front of you wants to eat the pie.

**john**:
No way! He can't have the pie.  I push him back.

**jason**:
Show him John!
That guy is dead meet!

**erik**:
Roll for initiative.
