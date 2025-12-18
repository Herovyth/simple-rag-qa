import gradio as gr


def create_ui(rag_pipeline):
    with gr.Blocks() as demo:
        gr.Markdown("""# RAG QA System""")

        gr.Markdown("""
                **RAG (Retrieval-Augmented Generation)** система дозволяє:
                - Для джерела даних було використано першу книгу Гаррі Поттера із датасету HuggingFace (https://huggingface.co/datasets/prasad3458/Harry_Potter_Books)
                - Відповідь на запитання генерується на основі знайденого контексту з першої книги.
                - Дані перед обробкою чанкуються.
                - Присутній ретрівер та реранкер.
                - Використовується LiteLLM із Groq.

                **Як користуватись:**
                1. Введіть ваш API ключ LLM
                2. Оберіть режим пошуку (`bm25`, `semantic`, `both` або `off`)
                3. Напишіть запитання
                4. Натисніть **Ask**
                """)

        api_key = gr.Textbox(
            label="LLM API Key",
            placeholder="Paste your API key here",
            type="password"
        )

        query = gr.Textbox(label="Your question")

        search_mode = gr.Radio(
            ["bm25", "semantic", "both", "off"],
            value="both",
            label="Retrieval mode"
        )

        ask_btn = gr.Button("Ask")

        answer = gr.Textbox(label="Answer")
        context = gr.Textbox(
            label="Retrieved context",
            lines=10
        )

        ask_btn.click(
            rag_pipeline,
            inputs=[query, search_mode, api_key],
            outputs=[answer, context]
        )

    return demo
