import gradio as gr
from .utils import process_file, generate_worklist, read_instructions


def iDotScheduler():
    with gr.Blocks(theme="default", css="footer {visibility: hidden} .container {max-width: 1500px; margin: auto}") as app:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(read_instructions())

            with gr.Column(scale=9):

                with gr.Row():
                    input_file = gr.File(label="Input Excel File")
                    output_folder = gr.Textbox(label="Output Folder")
                    plate_size = gr.Dropdown(choices=[96, 384, 1536], value=1536, label="Plate Size")
                    
                with gr.Row():
                    read_btn = gr.Button("Read Files")
                    generate_btn = gr.Button("Generate Worklist")

                with gr.Row():
                    output_msg = gr.Textbox(label="Status")



                with gr.Row():
                    legend = gr.HTML(label="Legend")

                with gr.Tabs():
                    with gr.TabItem("Source ID"):
                        table1 = gr.HTML()
                    with gr.TabItem("Source Vol"):
                        table2 = gr.HTML() 
                    with gr.TabItem("Target ID"):
                        table3 = gr.HTML()
                    with gr.TabItem("Target Vol"):
                        table4 = gr.HTML()
                        
                
                read_btn.click(
                    process_file,
                    inputs=[input_file],
                    outputs=[table1, table2, table3, table4, legend]
                )
                
                generate_btn.click(
                    generate_worklist,
                    inputs=[input_file, output_folder, plate_size],
                    outputs=[output_msg]
                )
    
    return app

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from .utils import process_file, generate_worklist, read_instructions
    
    app = iDotScheduler()
    app.launch()
