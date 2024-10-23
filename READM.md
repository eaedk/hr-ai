# Resume and Video Interview Processing Pipeline with Gradio App

## Overview

This project processes a resume (PDF) and an interview video (audio) using AI-powered text and audio analysis. The end-to-end pipeline can now be accessed through an easy-to-use Gradio app interface that allows users to upload their CV and video interview, process the files, and download the results in a structured JSON format.

### Key Features:

* **Resume Text Extraction**: Extracts and processes the text from a PDF resume using AI to create a structured CV.
* **Audio Analysis from Video**: Extracts audio from a video interview and computes features such as speech rate, pauses, average volume, and sentiment analysis.
* **Gradio Interface**: A simple drag-and-drop interface where users can upload their CV and video, visualize the output as JSON, and download the processed results.
* **JSON Download**: Results are saved as a JSON file with the CV's full name embedded in the filename.

## Gradio App Interface

### App Functionality:

1. **Upload CV (PDF)**: Drag and drop the CV in PDF format.
2. **Upload Video Interview**: Drag and drop the interview video file.
3. **Process Button**: Click to start processing the CV and video in parallel.
4. **View Output**: The results of the analysis are displayed in a JSON format.
5. **Download Output**: Download the structured results as a JSON file with the CV's full name in the filename.

### Example JSON Output:

The output includes structured data from the CV as well as features extracted from the interview video:

```json
{
  "structured_resume": {
    "full_name": "John Doe",
    "education": [
      {
        "degree": "Master of Science",
        "institution": "XYZ University",
        "year": "2020"
      }
    ],
    "experience": [...]
  },
  "video_audio_features": {
    "pause_count": 5,
    "total_pause_duration_in_seconds": 12.3,
    "speech_rate_wpm": 120,
    "average_volume": 0.52,
    "sentiment": "Positive"
  }
}
```

## Installation

1. **Clone the repository**:
   

```bash
   git clone https://github.com/eaedk/hr-ai.git
   cd hr-ai
   ```

2. **Install dependencies**:
   Make sure you have Python 3.10+ installed, then install the required packages:
   
```bash
   pip install -r requirements.txt
   ```

  PS: Read the file `install_ffmpeg` and follow the instructions to install `ffmpeg`. 

3. **Set up OpenAI API Key**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   

```plaintext
   API_KEY=your_openai_api_key
   ```

4. **Run the Gradio app**:
   Start the Gradio app to interact with the pipeline:
   

```bash
   python app.py
   ```

5. **Access the app**:
   Once the server starts, Gradio will provide a local URL (e.g., `http://127.0.0.1:7860` ) that you can open in your browser to access the drag-and-drop interface.

## Repository tree
Look at how the repository is structured.

```plaintext
.
├── .env # to be created after cloning the repo
├── README.md
├── app.py
├── img
│   ├── app_interface.png
│   └── launch_app.png
├── install_ffmpeg.txt
├── requirements.txt
├── src
│   ├── resume_interview_analyzer.py
│   └── structured_resume.py
└── storage
```

## How to Use the Gradio App

1. Open the Gradio app in your browser.
2. Drag and drop your PDF resume and interview video into the respective areas.
3. Click the "Process" button to run the pipeline.
4. View the structured output in the JSON section.
5. Download the JSON file using the provided button, with the file name automatically set to the CV's full name.

## Contribution

Don't hesitate to create some issues for helping me fixing the bugs.
