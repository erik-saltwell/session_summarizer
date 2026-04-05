# Make quality evaluation command for looking at diarization quality

# Threshold for low cosine similarity and improve identification logic using this (also add a flag we can use for later creating speaker clips)
- make the similarity a field on SpeechClips
- make a flag for low similarity, and a flag for high similarity
- put these thresholds as settings in session_settings (I think there is for low already)
- udpate create-speaker-clips to only use clips above a threshold cosine similarity
- when identifying speakers, leave identity None for low threshold, then compute centroids by speaker_id (use stabilization approach from research doc, starting with any clip where a speaker is one of the speakers)
- At this point, any clip has a number of similarities: the best clip match, and the best centroid match, use whichever has a higher cosine similarity.


# Segment based on TPST, not VAD
- make new TPST algo.
- also see if the TPST signal can be used in identifying end of turn

# Add diarizationlm suppport usig their HF model

# Add punctuation

# Final LLM cleanup call and write out final transcript
