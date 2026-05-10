<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Session Summarizer</h3>

  <p align="center">
    Transcribe Tabletop Roleplaying Game (TTRPG) sessions and generate handout-ready session summaries.
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started. To avoid retyping too much info, do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`, `project_license`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

#### 1. Install Ubuntu system packages

```bash
sudo apt update
sudo apt install -y libsndfile1 ffmpeg build-essential
uv sync
```

#### 2. Download the Smart Turn ONNX model

The [Smart Turn](https://github.com/pipecat-ai/smart-turn) model provides end-of-turn detection — it predicts whether a speaker has finished their conversational turn by analysing raw 16 kHz audio.  This project vendors the Smart Turn inference code locally (in `src/smart_turn/`) and loads the ONNX model from `models/smart-turn/`.

Download the model weights from HuggingFace ([pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3)):

```bash
# CPU version (8.7 MB, int8 quantized — recommended for most setups)
huggingface-cli download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx \
  --local-dir models/smart-turn

# GPU version (32.4 MB, fp32 — slightly more accurate, faster on GPU)
huggingface-cli download pipecat-ai/smart-turn-v3 smart-turn-v3.2-gpu.onnx \
  --local-dir models/smart-turn
```

If you don't have the HuggingFace CLI, install it first:

```bash
uv pip install huggingface-hub[cli]
```

The inference code will prefer the GPU model if both are present; otherwise it uses the CPU model.

**How it works:** Smart Turn uses Whisper Tiny as a backbone with a linear classifier (~8M parameters).  It accepts up to 8 seconds of 16 kHz mono PCM audio and returns an end-of-turn probability.  The project's `update-turn-end` command uses this to score each speech segment in a diarised recording.

> **Note on `LocalSmartTurnAnalyzerV3`:** This class is part of the [Pipecat](https://pipecat.ai) real-time voice agent framework (v0.0.85+) and wraps the same underlying model for live streaming pipelines.  For this project's offline/batch scoring pipeline, we call `predict_endpoint()` directly from the vendored `smart_turn` package instead of going through Pipecat.

#### 3. Install en_core_web_sm :
```bash
uv run python -m spacy download en_core_web_sm
```

#### 4. Google Cloud setup (for Google Chirp transcriber)

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and log in:

```bash
gcloud auth login
gcloud auth application-default login
```

This authenticates both the Speech-to-Text API and Google Cloud Storage (used for temporary audio uploads during batch transcription).

Set the required environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GCS_BUCKET="your-bucket-name"
```

Make sure your GCP project has the **Cloud Speech-to-Text API** enabled and the bucket exists:

```bash
gcloud services enable speech.googleapis.com --project=$GOOGLE_CLOUD_PROJECT
gcloud storage buckets create gs://$GCS_BUCKET --project=$GOOGLE_CLOUD_PROJECT --location=us
```

#### 5. OpenAI setup (for GPT-4o transcriber)

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

#### 6. Quick verification

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

import nemo.collections.asr as nemo_asr
print("NeMo ASR import OK")
PY
```



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Run any command via the CLI module:

```bash
uv run python -m session_summarizer <command> [options]
```

Most commands operate on a single session and require `--session/-s <session_id>`. A session is a directory under `data/<session_id>/` containing a `settings.yaml` and the source audio file. Pipeline commands write their outputs **into that session directory** at the paths declared by `settings.paths.*` (see `src/session_summarizer/settings/session_settings.py`). Commands that write outside the session directory are flagged below.

## Commands

### Pipeline commands (operate on `--session`)

These run as ordered pipeline steps. Each one re-runs upstream dependencies if their outputs are stale, and writes a single artifact (typically a `SpeechClipSet` JSON) into the session directory.

- **`clean-audio`** — Runs noise reduction on the source recording.
  Output: `<session_dir>/<paths.cleaned_audio>` (WAV).

- **`diarize-audio`** — Sends cleaned audio to ElevenLabs Scribe v2 to produce the initial diarized `SpeechClipSet`. For the `test` session it also enhances words with ground truth.
  Output: `<session_dir>/<paths.base_diarization>` (JSON).

- **`add-embeddings`** — Computes a speaker embedding vector for each speech clip.
  Output: `<session_dir>/<paths.clips_with_embeddings>` (JSON).

- **`identify-speakers`** — Matches each clip's embedding against attendee centroids in `registered_speakers.yaml` and assigns identities. Requires `attendees` in session settings and that each attendee is registered.
  Output: `<session_dir>/<paths.identified_speakers>` (JSON).

- **`apply-identity-stitching`** — Merges adjacent clips that share an identity into single utterances.
  Output: `<session_dir>/<paths.identity_stitched>` (JSON).

- **`mark-backchannels`** — Flags short acknowledgement clips ("uh huh", "yeah") as backchannels.
  Output: `<session_dir>/<paths.backchannel_marked>` (JSON).

- **`punctuate-text`** — Restores punctuation and capitalization on each clip's transcript.
  Output: `<session_dir>/<paths.punctuated_text>` (JSON).

- **`infer-speakers`** — Uses transcript text to infer role-based speaker identities (e.g., GM vs. player). Also writes an inferred-participants sidecar and patches the session's `settings.yaml` with the inferred mapping.
  Outputs: `<session_dir>/<paths.inferred_speakers>` (JSON), an `_inferred_participants` companion file alongside it, and an in-place update to `<session_dir>/settings.yaml`.

- **`assign-utterance-ids`** — Stamps each clip with a stable `<campaign_id>_<session_id>_<n>` utterance id.
  Output: `<session_dir>/<paths.utterance_ids_annotated>` (JSON).

- **`save-session-clipset`** — Saves the utterance-id-stamped clipset under the canonical session filename.
  Output: `<session_dir>/<session_id>.json`.

- **`simplify-transcript`** — Generates an LLM-cleaned narrative transcript from the punctuated clipset.
  Output: `<session_dir>/<paths.simplified_transcript>`.

- **`summarize-session`** — Generates the final session summary with Claude. The output filename has the session date appended (e.g. `summary_2026_05_09.md`).
  Output: `<session_dir>/<paths.summary_path>` with the session date appended to the filename.

- **`validate-diarization`** — Evaluates diarization quality (DER, JER, WDER) across pipeline stages and prints a comparison table. Console output only — does not write files.

### Session-management commands

- **`clean-session --session <id>`** — Deletes every file in `<session_dir>` except `settings.yaml` and the configured source audio. Affects only that session directory.

- **`clean-diarization --session <id>`** — Deletes outputs of selected upstream pipeline steps inside `<session_dir>` so they re-run on next invocation.

### Speaker registration commands

These manage the on-disk speaker library and write **outside the session directory**, under `voice_samples/` at the project root.

- **`create-known-speaker-clips --session <id> --temp-folder <name>`** — Extracts per-speaker audio clips for clips with confidently identified speakers.
  Output: WAV files written to `voice_samples/<temp_folder>/` (the temp folder is emptied first). This is **outside** the session directory.

- **`create-speaker-clips-from-inferred-speakers --session <id>`** — Extracts per-speaker audio clips using the inferred-speaker mapping.
  Output: WAV files written into the top-level per-speaker folders under `voice_samples/<speaker>/`. **Outside** the session directory.

- **`merge-speaker-clips --speaker <name> --output-folder <path>`** — Concatenates short clips for a speaker (with a small silence gap) until each merged clip meets `speaker_clips.min_duration_seconds`.
  Output: merged WAV files written to the user-supplied `--output-folder`.

- **`remove-outlier-speaker-clips --speaker <name> --output-folder <path>`** — Removes clips whose embedding is too far from the speaker's centroid.
  Output: filtered WAV files written to the user-supplied `--output-folder`.

- **`register-speakers`** — For every per-speaker folder in `voice_samples/`, merges + filters the clips, computes a centroid embedding, and registers it.
  Output: updates `voice_samples/registered_speakers.yaml` (project root, **outside** any session directory).

### Utility commands

- **`generate-sample-settings`** — Writes a documented sample settings file.
  Output: `data/settings.yaml` at the project root (**outside** any session directory).

- **`document-dependencies`** — Inspects all pipeline commands and emits a Mermaid file/command dependency graph.
  Output: `data/command_dependencies.mmd` at the project root (**outside** any session directory).

- **`clear-logs`** — Deletes every file in the project-level `logs/` directory (**outside** any session directory).

- **`test --session <id>`** — Developer scratch command for re-running an arbitrary command across a filtered set of sessions. Outputs depend on the inner command being driven.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
