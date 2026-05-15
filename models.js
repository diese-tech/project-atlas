// Central model registry — edit here to add/remove models app-wide
export const MODELS = [
  { value: "qwen2.5-coder:14b", label: "Qwen 2.5 Coder 14B" },
  { value: "qwen2.5:latest",    label: "Qwen 2.5 7B" },
  { value: "llama3:latest",     label: "Llama 3" },
  { value: "mistral:latest",    label: "Mistral" },
  { value: "phi3:latest",       label: "Phi-3" },
];
export const DEFAULT_MODEL = MODELS[0].value;

// Populates a <select> element with the model list
export function populateModelSelect(selectEl, selected = DEFAULT_MODEL) {
  selectEl.innerHTML = "";
  for (const m of MODELS) {
    const opt = document.createElement("option");
    opt.value = m.value;
    opt.textContent = m.label;
    if (m.value === selected) opt.selected = true;
    selectEl.appendChild(opt);
  }
}