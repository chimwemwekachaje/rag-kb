<!-- 3361bdd4-036a-4a15-ac46-93f68833822b 77cac00d-de78-4e28-a6fa-9bbe23cea521 -->
# Add Clear Button and Submit Handler

## Changes to `/Users/kachaje/andela/genAi-bootcamp/projects/code/rag-kb/app.py`

### 1. Add Clear Button

In the `launch_gradio_ui` function around line 506, after the Search button:

- Add a Clear button next to the Search button using `gr.Button("Clear", variant="secondary")`
- Place both buttons in a `gr.Row()` to display them side-by-side

### 2. Create Clear Function

Add a helper function `clear_inputs()` that returns empty strings to clear both:

- The query input field
- The query output field

### 3. Add Clear Button Event Handler

Connect the Clear button to the `clear_inputs()` function:

- Input: None
- Outputs: `[query_input, query_output]`

### 4. Add Submit Handler for Query Input

Add a `.submit()` event handler to the `query_input` TextArea:

- This triggers when user presses Enter
- Calls the same `rag_query` function as the Search button
- Input: `query_input`
- Output: `query_output`

## Implementation Details

The changes will be made around lines 501-516 in the UI creation section within the left column where the query components are defined.

```python
# Before (lines 501-507):
query_input = gr.TextArea(...)
query_button = gr.Button("Search", variant="primary")
query_output = gr.TextArea(...)

# After:
query_input = gr.TextArea(...)
with gr.Row():
    query_button = gr.Button("Search", variant="primary")
    clear_button = gr.Button("Clear", variant="secondary")
query_output = gr.TextArea(...)
```

Event handlers will be added in the section starting at line 513 alongside the existing event handlers.

### To-dos

- [ ] Add Clear button in a Row layout next to Search button
- [ ] Create clear_inputs() helper function to return empty strings
- [ ] Add event handlers for Clear button and query_input.submit()