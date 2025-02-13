import gradio as gr
import os
from .utils import read_instructions
from .ui_utils import visualise_input_data
from .core import generate_worklist
from .constants import ParallelisationType  # Add this import at the top
from .output_check_view import visualise_plate_output


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
                    parallelisation = gr.Dropdown(
                        choices=[p.value for p in ParallelisationType], 
                        value=ParallelisationType.IMPLICIT.value,
                        label="Parallelisation Strategy"
                    )
                with gr.Row():
                    source_plate_type = gr.Dropdown(
                        choices=["S.100 Plate", "S.200 Plate"], 
                        value="S.200 Plate", 
                        label="Source Plate Type"
                    )
                    source_name = gr.Textbox(
                        label="Source Plate Name", 
                        value="Source Plate 1"
                    )
                    target_type = gr.Dropdown(
                        choices=["MWP 96", "MWP 384", "1536_Screenstar"], 
                        value="MWP 96",
                        label="Target Type",
                        info="Target plate type"
                    )
                    target_name = gr.Textbox(
                        label="Target Name", 
                        value="Target Plate 1",
                        info="Name of the target plate"    
                    )
                with gr.Row():
                    advanced_settings = gr.Accordion("Advanced Settings", open=False)
                    with advanced_settings:
                        with gr.Row():
                            dispense_waste = gr.Checkbox(
                                label="Dispense to Waste", 
                                value=True,
                                info="Enable/disable priming before dispensing"
                            )
                            deionisation = gr.Checkbox(
                                label="Use Deionisation", 
                                value=True,
                                info="Enable/disable deionization of source and target plates (Should always be True)"
                            )
                            optimization = gr.Dropdown(
                                choices=["NoOptimization", "Reorder", "ReorderAndParallel"],
                                value="ReorderAndParallel", 
                                label="Optimization Level",
                                info="Protocol optimization to reduce total dispensing time"
                            )
                            debug_mode = gr.Checkbox(
                                label="Debug Mode", 
                                value=False,
                                info="Enable detailed logging"
                            )

                with gr.Row():
                    read_btn = gr.Button("Read Files")
                    generate_btn = gr.Button("Generate Worklist")

                with gr.Row():
                    output_msg = gr.Textbox(label="Status")
                    idot_wl = gr.State()  # Change back to State

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
                with gr.Tabs():
                    with gr.TabItem("Plate Check View"):
                        plate_view_html = gr.HTML()
    
                read_btn.click(
                    visualise_input_data,
                    inputs=[input_file],
                    outputs=[table1, table2, table3, table4, legend]
                )
                generate_btn.click(
                    fn=lambda *inputs: generate_worklist(
                        inputs[0],                       # input_file
                        inputs[1],                       # output_folder
                        inputs[2],                       # plate_size
                        ParallelisationType(inputs[3]),  # parallelisation
                        inputs[13],                      # debug_mode (new)
                        {
                            'worklist_name': inputs[4],
                            'user': inputs[5],
                            'source_plate_type': inputs[6],
                            'source_name': inputs[7],
                            'target_type': inputs[8],
                            'target_name': inputs[9],
                            'dispense_to_waste': inputs[10],
                            'deionisation': inputs[11],
                            'optimization': inputs[12]
                        }
                    ),
                    inputs=[
                        input_file, output_folder, plate_size, parallelisation, worklist_name, user,
                        source_plate_type, source_name, target_type, target_name,
                        dispense_waste, deionisation, optimization, debug_mode
                    ],
                    outputs=[output_msg, idot_wl]
                )
            idot_wl.change(
                fn=visualise_plate_output,
                inputs=[idot_wl, plate_size],
                outputs=[plate_view_html]
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
