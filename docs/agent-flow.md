# How the agent works

Three views of `src/huggingface_agent/app.py`: who talks to whom, the loop the agent repeats, and one real request.

The model never runs anything itself. It writes Python, the local machine runs it, and the result goes back to the model.

## 1. Parts and who talks to whom

```mermaid
%%{init: {'theme':'neutral', 'flowchart': {'curve': 'basis'}}}%%
flowchart LR
  U(["You<br/>in the browser"]) -->|"chat message"| G["Gradio page<br/>Gradio_UI.py"]
  G -->|"task"| A["CodeAgent<br/>app.py"]
  A -->|"task + tool descriptions"| M[["Qwen2.5-Coder-32B<br/>runs on Hugging Face"]]
  M -->|"Thought + Python code"| A
  A -->|"runs the code locally"| T
  subgraph T["Tools the code can call"]
    direction TB
    t1["get_current_time_in_timezone"]
    t2["image_generator"]
    t3["DuckDuckGoSearchTool"]
    t4["final_answer"]
  end
  T -->|"result = Observation"| A
  A -->|"final answer"| G
  G -->|"text or image"| U
  ENV[".env<br/>HF_TOKEN"] -.->|"authorizes calls"| M

  classDef you fill:#dff1ee,stroke:#0c7c72,color:#14212b
  classDef brain fill:#fdebc8,stroke:#b7791f,color:#14212b
  classDef worker fill:#dbe7f5,stroke:#3b6ea5,color:#14212b
  classDef tool fill:#ffffff,stroke:#55656f,color:#14212b
  class U,G you
  class M brain
  class A worker
  class t1,t2,t3,t4 tool
```

Green: what you see. Blue: the agent that does the work. Amber: the model, which only thinks. White: the tools it may call. The dotted line is the Hugging Face token from `.env`.

## 2. The loop inside one question

The agent repeats think, code, run, observe until the code calls `final_answer`, or until it hits `max_steps=6`.

```mermaid
%%{init: {'theme':'neutral', 'flowchart': {'curve': 'basis'}}}%%
flowchart TD
  Q(["Your question"]) --> P["Agent sends the question<br/>and tool list to the model"]
  P --> TH["Model writes a Thought<br/>and Python code"]
  TH --> R["Agent runs that code<br/>on your machine"]
  R --> C{"Did the code call<br/>final_answer?"}
  C -->|"yes"| F(["Answer shown in the chat"])
  C -->|"no, or the code failed"| O["Printed output or error<br/>becomes an Observation"]
  O --> S{"Steps used<br/>less than 6?"}
  S -->|"yes"| TH
  S -->|"no"| X(["Stops without an answer"])

  classDef done fill:#dff1ee,stroke:#0c7c72,color:#14212b
  classDef stop fill:#f9dcd8,stroke:#b4443a,color:#14212b
  classDef step fill:#dbe7f5,stroke:#3b6ea5,color:#14212b
  classDef ask fill:#fdebc8,stroke:#b7791f,color:#14212b
  class F,Q done
  class X stop
  class P,TH,R,O step
  class C,S ask
```

Errors are not fatal: a failed run is fed back as an Observation, so the model gets another try until the step limit.

## 3. One real request: "Draw a dog"

```mermaid
%%{init: {'theme':'neutral'}}%%
sequenceDiagram
  autonumber
  actor You
  participant UI as Gradio page
  participant Agent as CodeAgent
  participant LLM as Qwen model
  participant Tool as image_generator
  participant Disk as outputs folder
  You->>UI: Draw a dog
  UI->>Agent: task
  Agent->>LLM: task and tool descriptions
  LLM-->>Agent: Python code that calls image_generator then final_answer
  Agent->>Tool: run image_generator with a prompt
  Tool->>Disk: save image-date-time.png
  Tool-->>Agent: image object
  Agent-->>UI: final answer is an image
  UI-->>You: picture appears in the chat
```

The picture is made by the tool, not by the model. A copy is saved to `outputs/` with a timestamp.

## Where each part lives

| Part | File or setting | Job |
|---|---|---|
| Chat page | `Gradio_UI.py` | Shows steps, text and images at `127.0.0.1:7860` |
| Agent and tools | `app.py` | Creates the `CodeAgent`, defines the tools, loads the image tool |
| Model | `InferenceClientModel` | Calls Qwen2.5-Coder-32B on Hugging Face |
| Token | `.env` → `HF_TOKEN` | Read by `load_dotenv()`, used automatically by the model client |
| Step limit | `max_steps=6` | Stops the loop so it cannot run forever |
| Saved images | `outputs/` | Every generated image is copied here with a timestamp |
