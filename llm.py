import litellm


def generate_answer(query, context_chunks, api_key):
    context = "\n\n".join(context_chunks)

    prompt = f"""
        Answer the question only in English using ONLY the context below.
        If the answer is not present, say "I thought we are talking about Harry Potter book".

        Context:
        {context}
        
        Question:
        {query}
        """

    response = litellm.completion(
        model="groq/llama-3.3-70b-versatile",
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}]

    )

    return response["choices"][0]["message"]["content"]
