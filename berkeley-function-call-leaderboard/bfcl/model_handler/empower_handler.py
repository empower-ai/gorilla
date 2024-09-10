from bfcl.model_handler.oss_handler import OSSHandler
from bfcl.model_handler.model_style import ModelStyle
import json
from bfcl.model_handler.utils import (
    convert_to_tool,
)
from bfcl.model_handler.constant import (
    GORILLA_TO_OPENAPI,
)


class EmpowerHandler(OSSHandler):
    def __init__(self, model_name, temperature=0.001, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.model_style = ModelStyle.OSSMODEL

    def _format_prompt(prompts, functions, test_category):
        formatted_prompt = "<|begin_of_text|>"

        for idx, prompt in enumerate(prompts):
            if idx == 0:
                tools = convert_to_tool(
                    functions, GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL, test_category
                )
                prompt['content'] = "In this environment you have access to a set of functions defined in the JSON format you can use to address user's requests, use them if needed.\nFunctions:\n" \
                    + json.dumps(tools, indent=2) \
                    + "\n\n" \
                    + "User Message:\n" \
                    + prompt['content']

            formatted_prompt += f"<|start_header_id|>{prompt['role']}<|end_header_id|>\n\n{prompt['content']}<|eot_id|>"

        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt

    def inference(
        self,
        test_question,
        num_gpus,
        gpu_memory_utilization,
        format_prompt_func=_format_prompt,
    ):
        return super().inference(
            test_question,
            num_gpus,
            gpu_memory_utilization,
            format_prompt_func=format_prompt_func,
            include_system_prompt=False,
        )

    def decode_ast(self, result, language="Python"):
        decoded_output = []

        # strip the function/conversation tag <f>/<c>
        result_stripped = result[3:]
        for invoked_function in json.loads(result_stripped):
            name = invoked_function["name"]
            params = invoked_function["arguments"]
            decoded_output.append({name: params})

        return decoded_output

    def decode_execute(self, result):
        execution_list = []

        for function_call in self.decode_ast(result):
            for key, value in function_call.items():
                argument_list = []
                for k, v in value.items():
                    argument_list.append(f'{k}={repr(v)}')
                execution_list.append(
                    f"{key}({','.join(argument_list)})"
                )

        return execution_list
