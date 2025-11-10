import google.generativeai as genai
genai.configure(api_key="AIzaSyCXuA_n7S3Ubv0rcBdiFfAj8D6M092B540")

def answer_query_with_context(query: str, chunks: list[str], format_json: bool = True) -> str:
    context = "\n\n".join(chunks)

    prompt = f"""
You are a legal assistant AI. A user is asking a question based on a legal document.
Here is the relevant content extracted from the document:

--- START OF CONTEXT ---
{context}
--- END OF CONTEXT ---

Now answer the following question based only on the above context:
"{query}"

{"Respond strictly in JSON format as 'answer', 'source_clause_excerpt'" if format_json else "Respond naturally."}
"""

    model = genai.GenerativeModel('gemini-2.0-flash-001')
    response = model.generate_content(prompt)
    return response.text


