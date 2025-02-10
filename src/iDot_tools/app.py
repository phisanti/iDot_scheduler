import gradio as gr
import os
from .utils import process_file, generate_worklist, read_instructions


def iDotScheduler():
    with gr.Blocks(theme="default", css="footer {visibility: hidden} .container {max-width: 1500px; margin: auto}") as app:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(read_instructions())

            with gr.Column(scale=9):

                with gr.Row():
                    input_file = gr.File(label="Input Excel File")
                    output_folder = gr.Textbox(label="Output Folder", value=os.getcwd())
                
                with gr.Row():
                    worklist_name = gr.Textbox(label="Worklist Name", value="iDot Worklist")
                    user = gr.Textbox(label="User Name", value="John Doe")
                    plate_size = gr.Dropdown(choices=[96, 384, 1536], value=1536, label="Plate Size")

                with gr.Row():
                    source_plate_type = gr.Dropdown(choices=["S.100 Plate", "S.200 Plate"], value="S.200 Plate", label="Source Plate Type")
                    source_name = gr.Textbox(label="Source Plate Name", value="Source Plate 1")
                    target_type = gr.Textbox(label="Target Type", value="1536_Screenstar")
                    target_name = gr.Textbox(label="Target Name", value="Target Plate 1")

                with gr.Row():
                    dispense_waste = gr.Checkbox(label="Dispense to Waste", value=True)
                    deionisation = gr.Checkbox(label="Use Deionisation", value=True)
                    optimization = gr.Dropdown(choices=["ReorderAndParallel", "None"], value="ReorderAndParallel", label="Optimization")

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
                    fn=lambda *inputs: generate_worklist(
                        inputs[0],  # input_file
                        inputs[1],  # output_folder
                        inputs[2],  # plate_size
                        {
                            'worklist_name': inputs[3],
                            'user': inputs[4],
                            'source_plate_type': inputs[5],
                            'source_name': inputs[6],
                            'target_type': inputs[7],
                            'target_name': inputs[8],
                            'dispense_to_waste': inputs[9],
                            'deionisation': inputs[10],
                            'optimization': inputs[11]
                        }
                    ),
                    inputs=[
                        input_file, output_folder, plate_size, worklist_name, user,
                        source_plate_type, source_name, target_type, target_name,
                        dispense_waste, deionisation, optimization
                    ],
                    outputs=[output_msg]
                )
                    
    return app

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from .utils import process_file, generate_worklist, read_instructions
    
    app = iDotScheduler()
    app.launch(
                inbrowser=True,  
                share=False      
    )
