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

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

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
