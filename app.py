import gradio as gr
import json
from src.resume_interview_analyzer import ResumeVideoPipeline
import os


# Function to run the pipeline and return results
def process_cv_and_interview(cv_file, video_file):
    # Saving uploaded files
    cv_path = cv_file.name
    video_path = video_file.name

    # Initialize the pipeline
    pipeline = ResumeVideoPipeline(cv_path, video_path)

    # Run the pipeline
    results_ = pipeline.run_pipeline()

    results = {'structured_resume':results_['structured_resume'].dict(),
                   'video_audio_features':results_['video_audio_features']}

    # Save the results as a JSON file with the CV's full name
    resume_result = results['structured_resume']
    
    # Assuming the 'full_name' field exists in the resume result
    full_name = resume_result.get('full_name', 'output').replace(" ", "_")
    output_filename = f"{full_name}_output.json"
    
    # Write the results to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)

    
    return results, output_filename

# Gradio UI
def gradio_interface():
    # UI Components
    cv_input = gr.File(label="Upload CV (PDF)", type="filepath")
    video_input = gr.File(label="Upload Video Interview", type="filepath")
    json_output = gr.JSON(label="Processed Output")
    download_output = gr.File(label="Download Output JSON")

    # Function to process the inputs and return the processed output and download link
    def process_files(cv, video):
        if cv is not None and video is not None:
            results, output_filename = process_cv_and_interview(cv, video)
            return results, output_filename

    # Interface layout
    with gr.Blocks() as demo:
        gr.Markdown("# CV and Interview Processor")
        
        with gr.Row():
            cv_input.render()
            video_input.render()

        process_btn = gr.Button("Process")

        json_output.render()
        download_output.render()

        # Button click triggers the processing and returns the output
        process_btn.click(process_files, [cv_input, video_input], [json_output, download_output])

    return demo

# Launch the Gradio app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch()
