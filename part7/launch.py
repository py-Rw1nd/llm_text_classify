import gradio as gr
from typing import List, Tuple
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import json
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained('autodl-tmp/dpo_qwen_plus_0710', trust_remote_code=True, padding_side='left')
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    'autodl-tmp/merge_model',
    device_map='auto',
    torch_dtype=torch.bfloat16,
    # cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage = True
)

model = PeftModel.from_pretrained(model, model_id='autodl-tmp/dpo_qwen_plus_0710',torch_type=torch.bfloat16)

## Input the prompt 
prompt_role = "你是一个语义分类器。"
prompt_goal = "以下是对卷烟产品的评论，按照分类标签语义进行分类，不需要说明理由。"
prompt_context = "可可的香味很迷人，烟气入后还算柔和，性价比高。"
prompt_expectations = "输出格式：'分类标签'：'A正面，B负面，C中性，D讽刺，E无产品评价语意'"


# function to clear the conversation
def reset() -> List:
    return []

# function to call the model to generate
def interact_summarization(role_textbox:str, goal_textbox:str,context_textbox:str ,expectations_textbox:str, temp = 0.3) -> List[Tuple[str, str]]:

    inputs = f"{role_textbox}\n{goal_textbox}+{context_textbox}+{expectations_textbox}"
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": inputs}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt'

    )
    generated_ids = model.generate(
        text.input_ids,
        # max_new_tokens=512,
        max_new_tokens=64,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
        do_sample = True
        
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(text.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return [response.text]


# This part constructs the Gradio UI interface
with gr.Blocks() as demo:
    gr.Markdown("# 评论文本分类系统\n本模型被训练用于完成多维度的文本分类!!")
    with gr.Row():
        with gr.Column():
            
            role_textbox = gr.Textbox(label="role", value=prompt_role,interactive=True)
            goal_textbox = gr.Textbox(label="goal", value = prompt_goal,interactive=True)
            context_textbox = gr.Textbox(label="context", value=prompt_context,interactive=True)
            expectations_textbox = gr.Textbox(label="expectations", value=prompt_expectations,interactive=True)
            temperature_slider = gr.Slider(0.0, 1.0, 0.3, step = 0.1, label="Temperature")
        with gr.Column():
            # gr.Markdown("#  Temperature\n Temperature is used to control the output of the chatbot. The higher the temperature is, the more creative response you will get.")
            
            chatbot = gr.Chatbot()
    with gr.Row():
        sent_button = gr.Button(value="Send")
        reset_button = gr.Button(value="Reset")

    # with gr.Column():
    #     gr.Markdown("#  Save your Result.\n After you get a satisfied result. Click the export button to recode it.")
    #     export_button = gr.Button(value="Export")
    sent_button.click(interact_summarization, inputs=[role_textbox, goal_textbox,context_textbox,expectations_textbox, temperature_slider], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    


demo.launch(share=True)