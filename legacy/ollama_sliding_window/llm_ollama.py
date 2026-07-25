"""
llm_ollama.py - Ollama-backed SLM forecasters for the legacy prototype.
"""
import re
import time
import ollama
from .base import BaseForecaster

PROMPT_CLINICAL = """Given these NIMAP blood pressure values: {history_text}
What are the next {n_steps} values? Reply with ONLY numbers, one per line, nothing else."""


class OllamaLLMForecaster(BaseForecaster):
    def __init__(self, model_name, display_name=None,
                 prompt_template=PROMPT_CLINICAL, temperature=0.0):
        self.model_name = model_name
        self.name = display_name or model_name
        self.prompt_template = prompt_template
        self.temperature = temperature

    def _format_history(self, history_times, history_values):
        return str([round(float(v), 1) for v in history_values])

    def _parse_output(self, text, n_steps):
        values = []
        for line in text.split('\n'):
            numbers = re.findall(r'\b(\d+\.?\d*)\b', line)
            for n in numbers:
                val = float(n)
                if 30 < val < 200:
                    values.append(val)
                    break
        return values[:n_steps]

    def predict(self, history_times, history_values, n_steps):
        history_text = self._format_history(history_times, history_values)
        prompt = self.prompt_template.format(
            history_text=history_text,
            n_steps=n_steps
        )

        max_retries = 3
        raw, last_error = None, None
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': self.temperature},
                )
                raw = response['message']['content'].strip()
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"    [RETRY {attempt+1}] {type(e).__name__}, waiting 3s...")
                    time.sleep(3)
                else:
                    print(f"    [ERROR] {self.name} failed: {e}")

        if raw is None:
            return [float(history_values[-1])] * n_steps, {
                'parse_failed': True, 'error': str(last_error),
                'raw_output': f'ERROR: {last_error}'
            }

        pred = self._parse_output(raw, n_steps)
        parse_failed = len(pred) < n_steps
        while len(pred) < n_steps:
            pred.append(float(history_values[-1]))

        return pred, {'parse_failed': parse_failed, 'raw_output': raw}


def Gemma3_4B():    return OllamaLLMForecaster('gemma3:4b', 'Gemma3-4B')
def Llama32_3B():   return OllamaLLMForecaster('llama3.2:3b', 'Llama3.2-3B')
def Phi35_Mini():   return OllamaLLMForecaster('phi3.5:latest', 'Phi3.5-Mini')
def Qwen25_3B():    return OllamaLLMForecaster('qwen2.5:3b', 'Qwen2.5-3B')
def Gemma2_2B():    return OllamaLLMForecaster('gemma2:2b', 'Gemma2-2B')
def Qwen25_15B():   return OllamaLLMForecaster('qwen2.5:1.5b', 'Qwen2.5-1.5B')
