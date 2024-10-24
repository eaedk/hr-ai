import os
import PyPDF2
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import instructor
from openai import OpenAI  
import librosa
import logging
import numpy as np
from pydub import AudioSegment
import whisper
from textblob import TextBlob
from src.structured_resume import CandidateProfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Resume Text Extraction and AI Processing
class ResumeProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = ""

    def extract_text_from_pdf(self):
        """
        Extract text from the PDF resume file.
        """
        logging.info(f"Extracting text from resume: {self.pdf_path}")
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    self.text += page.extract_text()
            logging.info(f"Text extraction successful for: {self.pdf_path}")
        except Exception as e:
            logging.error(f"Error reading {self.pdf_path}: {e}")
        return self.text

    def process_resume_with_ai(self):
        """
        Process the resume text using OpenAI to create structured resume content.
        """
        load_dotenv()
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("API key not found in the .env file.")
        
        logging.info("Processing resume with AI...")
        client = instructor.from_openai(OpenAI(api_key=api_key), mode=instructor.Mode.JSON)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": self.text}],
            response_model=CandidateProfile  # Assuming CandidateProfile is pre-defined
        )
        logging.info("Resume processing completed with AI.")
        return response


# Video Processing with Audio Feature Extraction
import os
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import whisper
from textblob import TextBlob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    def __init__(self, video_path, output_folder=None, noise_factor=1.0, min_pause_duration=0.3, model_name="turbo"):
        """
        Initialize the VideoProcessor class with the video path and optional output folder.
        
        Parameters:
        video_path (str): Path to the video file.
        output_folder (str): Path to the folder where the audio and segments will be saved. Defaults to current working directory.
        noise_factor (float): Factor to adjust noise threshold for dynamic environments. Default is 1.0.
        min_pause_duration (float): Minimum duration of a pause to be considered silence. Default is 0.3 seconds.
        model_name (str): Name of the Whisper model to use for transcription. Default is "turbo".
        """
        self.video_path = video_path
        self.output_folder = output_folder if output_folder else os.getcwd()
        self.noise_factor = noise_factor
        self.min_pause_duration = min_pause_duration
        self.model_name = model_name
        self.audio_filepath = self.generate_audio_filename()
        

    def generate_audio_filename(self):
        """
        Generate the audio filename based on the video filename.
        
        Returns:
        str: Path to the audio file.
        """
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(self.output_folder, f"{base_name}.wav")

    def extract_audio_from_video(self):
        """
        Extract audio from the video and save it as a .wav file.
        """
        logging.info(f"Extracting audio from video: {self.video_path}")
        try:
            video = VideoFileClip(self.video_path)
            audio = video.audio
            audio.write_audiofile(self.audio_filepath)
            logging.info(f"Audio extracted and saved to: {self.audio_filepath}")
        except Exception as e:
            logging.error(f"Error extracting audio from {self.video_path}: {e}")
        return self.audio_filepath

    def compute_energy_db(self, y, frame_length=2048, hop_length=512):
        """
        Compute the energy of an audio signal in decibels.
        
        Parameters:
        y (np.ndarray): Audio time series.
        frame_length (int): Length of the analysis frame. Default is 2048.
        hop_length (int): Number of audio samples between successive frames. Default is 512.
        
        Returns:
        np.ndarray: Energy of the audio in decibels.
        """
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)
        return energy_db

    def compute_dynamic_threshold(self, energy_db):
        """
        Compute the dynamic threshold for silence detection based on the energy of the audio.
        
        Parameters:
        energy_db (np.ndarray): Energy of the audio in decibels.
        
        Returns:
        float: Dynamic threshold in decibels.
        """
        median_db = np.median(energy_db)
        std_db = np.std(energy_db)
        return median_db - (std_db * self.noise_factor)

    def extract_non_silence_frames(self, y, threshold_db):
        """
        Identify non-silence segments in the audio based on the dynamic threshold.
        
        Parameters:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.
        threshold_db (float): Dynamic threshold in decibels.
        
        Returns:
        np.ndarray: Array of start and end frames of non-silence segments.
        """
        return librosa.effects.split(y, top_db=-threshold_db)

    def process_silence_segments(self, non_silence_frames, original_audio, sr):
        """
        Process the audio to separate silence and non-silence segments.
        
        Parameters:
        non_silence_frames (np.ndarray): Array of start and end frames of non-silence segments.
        original_audio (AudioSegment): Original audio loaded as a pydub AudioSegment object.
        sr (int): Sampling rate of the audio.
        
        Returns:
        tuple: (pause_count, total_pause_duration, silence_segments, non_silence_segments)
        """
        silence_segments = AudioSegment.silent(duration=0)
        non_silence_segments = AudioSegment.silent(duration=0)

        total_duration_ms = len(original_audio)
        last_end_ms = 0
        total_pause_duration = 0
        pause_count = 0

        for start_frame, end_frame in non_silence_frames:
            start_ms = (start_frame / sr) * 1000
            end_ms = (end_frame / sr) * 1000
            non_silence_segments += original_audio[start_ms:end_ms]

            if start_ms > last_end_ms:
                silence_duration = (start_ms - last_end_ms) / 1000
                if silence_duration >= self.min_pause_duration:
                    pause_count += 1
                    total_pause_duration += silence_duration
                    silence_segments += original_audio[last_end_ms:start_ms]

            last_end_ms = end_ms

        if last_end_ms < total_duration_ms:
            final_silence_duration = (total_duration_ms - last_end_ms) / 1000
            if final_silence_duration >= self.min_pause_duration:
                pause_count += 1
                total_pause_duration += final_silence_duration
                silence_segments += original_audio[last_end_ms:total_duration_ms]

        return pause_count, total_pause_duration, silence_segments, non_silence_segments

    def save_audio_segments(self, silence_segments, non_silence_segments):
        """
        Save the processed silence and non-silence segments as separate audio files.
        
        Returns:
        tuple: (silence_output_file, non_silence_output_file)
        """
        base_name = os.path.splitext(os.path.basename(self.audio_filepath))[0]
        silence_output_file = os.path.join(self.output_folder, f"{base_name}_silence.wav")
        non_silence_output_file = os.path.join(self.output_folder, f"{base_name}_non_silence.wav")

        silence_segments.export(silence_output_file, format="wav")
        non_silence_segments.export(non_silence_output_file, format="wav")

        return silence_output_file, non_silence_output_file

    # Function to compute speech rate in words per minute
    def compute_speech_rate(self, transcription, y, sr):
        """
        Calculate the speech rate (words per minute).

        Parameters:
        transcription (str): Transcription of the audio text.
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.

        Returns:
        float: Speech rate in words per minute.
        """
        word_count = len(transcription.split())
        duration_minutes = librosa.get_duration(y=y, sr=sr) / 60
        return word_count / duration_minutes

    # Function to compute the average volume of the audio
    def compute_average_volume(self, y):
        """
        Calculate the average volume (RMS) of the audio signal.

        Parameters:
        y (np.ndarray): Audio time series.

        Returns:
        float: Average volume (RMS).
        """
        rms = librosa.feature.rms(y=y)
        return float(np.mean(rms))

    # Function to transcribe audio to text
    def transcribe_audio(self, audio_filepath, model_name="turbo"):
        """
        Transcribe audio into text using the Whisper model.

        Parameters:
        audio_filepath (str): Path to the audio file.
        model_name (str): Name of the audio transcription model.

        Returns:
        str: Transcribed text.
        """
        logging.info(f"Transcribing audio file: {audio_filepath}.")
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_filepath)

        logging.info(f"Audio file transcribed.")

        transcription = result['text']
        return transcription

    # Method to perform sentiment analysis on the transcription
    def sentiment_analysis(self, text):
        """
        Perform sentiment analysis on the transcribed text.
        
        Returns:
        str: Sentiment result - "Positive", "Neutral", or "Negative".
        """
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return "Positive"
        elif analysis.sentiment.polarity == 0:
            return "Neutral"
        else:
            return "Negative"
        
    def extract_additional_audio_features(self, y, sr):
        """
        Extract additional features from the audio data.
        
        Returns:
        tuple: (speech_rate_wpm, avg_volume, sentiment)
        """
        
        # Compute speech rate and average volume
        speech_rate_wpm = self.compute_speech_rate(self.transcription, y, sr)
        avg_volume = self.compute_average_volume(y)
        # Perform sentiment analysis on the transcription
        sentiment = self.sentiment_analysis(self.transcription)

        return speech_rate_wpm, avg_volume, sentiment

    def extract_audio_features(self):
        """
        Extract features from the audio file and process it into silence/non-silence segments.
        """
        logging.info(f"Loading audio file for analysis: {self.audio_filepath}")
        try:
            
            y, sr = librosa.load(self.audio_filepath)
            energy_db = self.compute_energy_db(y)
            threshold_db = self.compute_dynamic_threshold(energy_db)

            non_silence_frames = self.extract_non_silence_frames(y, threshold_db)
            original_audio = AudioSegment.from_wav(self.audio_filepath)

            pause_count, total_pause_duration, silence_segments, non_silence_segments = self.process_silence_segments(
                non_silence_frames, original_audio, sr
            )

            silence_output_file, non_silence_output_file = self.save_audio_segments(silence_segments, non_silence_segments)
            logging.info(f"Audio features extracted and segments saved: {silence_output_file} , {non_silence_output_file}.")

            # Transcribe the audio # Maybe we will use non_silence_output_file later
            self.transcription = self.transcribe_audio(self.audio_filepath, self.model_name)
            speech_rate_wpm, avg_volume, sentiment = self.extract_additional_audio_features(y, sr)
            logging.info(f"Audio features extracted and segments saved: {silence_output_file} , {non_silence_output_file}.")

            return {
                "pause_count": pause_count,
                "total_pause_duration_in_seconds": total_pause_duration,
                "speech_rate_wpm": speech_rate_wpm, 
                "average_volume": avg_volume, 
                "sentiment": sentiment,
                # "silence_output_file": silence_output_file,
                # "non_silence_output_file": non_silence_output_file
            }
        except Exception as e:
            logging.error(f"Error processing audio features from {self.audio_filepath}: {e}")
            return None

# # Example usage
# if __name__ == "__main__":
#     video_file = "path_to_video.mp4"
#     processor = VideoProcessor(video_file)
#     processor.extract_audio_from_video()
#     features = processor.extract_audio_features()
#     print(features)

# End-to-End Pipeline Class
class ResumeVideoPipeline:
    def __init__(self, resume_pdf_path, video_path):
        """
        Initialize the ResumeVideoPipeline class with paths to the resume PDF and video files.
        
        Parameters:
        resume_pdf_path (str): Path to the resume PDF file.
        video_path (str): Path to the video interview file.
        """
        self.resume_processor = ResumeProcessor(resume_pdf_path)
        self.video_processor = VideoProcessor(video_path)

    def run_pipeline(self):
        """
        Run the end-to-end pipeline for processing the resume and the video in parallel.
        """
        logging.info("Starting end-to-end pipeline...")
        
        with ThreadPoolExecutor() as executor:
            # Schedule the resume and video processing tasks to run in parallel
            future_resume = executor.submit(self.process_resume_pipeline)
            future_video = executor.submit(self.process_video_pipeline)
            
            # Wait for both tasks to complete and get their results
            resume_result = future_resume.result()
            video_features = future_video.result()

        logging.info("End-to-end pipeline completed.")
        return {
            "structured_resume": resume_result,
            "video_audio_features": video_features
        }

    def process_resume_pipeline(self):
        """
        Process the resume by extracting text and sending it to the AI for processing.
        """
        resume_text = self.resume_processor.extract_text_from_pdf()
        return self.resume_processor.process_resume_with_ai()

    def process_video_pipeline(self):
        """
        Process the video by extracting audio and performing feature extraction.
        """
        self.video_processor.extract_audio_from_video()
        return self.video_processor.extract_audio_features()


# # Example Usage
# if __name__ == "__main__":
#     # Define paths to the resume and video files
#     resume_pdf = "path_to_resume.pdf"
#     video_file = "path_to_video_interview.mp4"
    
#     # Initialize the end-to-end pipeline
#     pipeline = ResumeVideoPipeline(resume_pdf, video_file)
    
#     # Run the pipeline
#     results = pipeline.run_pipeline()

#     # Print the results
#     logging.info(f"Pipeline Results: {results}")
