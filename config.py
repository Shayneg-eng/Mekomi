"""
Configuration file for AI-Driven Document Analysis and Query Matching System
Contains all prompts, constants, and configurable parameters
"""

# Directory Configuration - these stay uppercase as they're constants
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

# Model Configuration - constants for model names
DESCRIPTION_MODEL = "gemma3n:e2b"
EMBEDDING_MODEL = "nomic-embed-text"
ANSWER_MODEL = "gemma3n:e4b"

# Processing Parameters - configuration values that control behavior
MIN_SENTENCE_LENGTH = 10
SLIDING_WINDOW_SIZE = 7
TARGET_SENTENCES = 5
MAX_QUERY_VARIATIONS = 3

# Performance Settings - tuning parameters for processing speed
PROGRESS_UPDATE_INTERVAL = 1000  # Show progress every N sentences
SAMPLE_SENTENCES_FIRST = 5     # First N sentences for document description
SAMPLE_SENTENCES_INTERVAL = 100  # Every Nth sentence for document description

# File Extensions - list of supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.txt']

# LLM Prompts - all the prompt templates we use for different tasks
PROMPTS = {
    'document_description': {
        'system': (
            "You are an expert in document analysis. Describe documents clearly, focusing on main topics and content."
        ),
        'assistant': (
            "Provide a concise 2-3 sentence summary of the document's key topics, themes, and purpose."
        ),
        'user_template': (
            "Document name: {document_name}\n\n"
            "Sentences 1-{first_count} (sample text):\n"
            "{sample_text}\n\n"
            "Task: Summarize the main topics, themes, and purpose of this document in 2-3 sentences."
        )
    },

    'query_preprocessing': {
        'system': (
            "You clean and standardize user queries. If splitting a multi-part query, ensure each part keeps ALL relevant info from the original so nothing is lost or ambiguous."
        ),
        'user_template': (
            "Analyze the user query below:\n"
            "1. Split into separate queries ONLY if the user query had multiple questions or parts.\n"
            "2. Correct grammar and spelling.\n"
            "3. Return a JSON array of cleaned queries.\n\n"
            "Examples:\n"
            "Input: 'Why does the sun shine so brightly that we can't look at it?'\n"
            "Output: [\"Why is the sun so bright?\", \"Why can't we look at the sun?\"]\n\n"
            "Input: 'When is the test and what chapters will it cover and do we need calculators?'\n"
            "Output: [\"When is the test?\", \"What chapters will the test cover?\", \"Do we need calculators for the test?\"]\n\n"
            "User query: '{user_query}'\n"
            "Return ONLY the JSON array.\n"
            "###IMPORTANT### make sure that if you split the user query, each new query has ALL relevant information from the original so each can stand alone."
            
        )
    },

    'query_variations': {
        'user_template': (
            "Generate {max_variations} diverse variations of the following search query: \"{user_query}\". Each variation must be semantically similar but use different phrasing."
        ),
        'assistant': (
            "Here are {max_variations} variations of your query for search robustness:"
        ),
        'instruction': (
            "Provide exactly {max_variations} query variations, each on a new line. Use synonyms and rephrasing, as if they might appear in a textbook. No numbering or extra text."
        )
    },

    'final_answer': {
        'context_intro': (
            "Below is the most relevant context to answer the user's question:"
        ),
        'user_template': (
            "Guide the student to the most relevant pages and sources for their question. Give a brief explanation, but do NOT provide direct answers. If the question is about a syllabus or schedule, you may answer directly.\n"
            "When referencing sources, smoothly cite them in-text, e.g. \"You should be able to find more about that on page 233.\" or \"To learn more about this, turn to page 253.\" List sources and context below.\n\n"
            "Sources List:\n<source_name> (Page <page_number>)\n\n"
            "Context:\n{context}\n\n"
            "Question: {user_query}"
        )
    },

    'final_answer_multiple': {
        'context_intro': (
            "Below is the most relevant context to answer the user's questions:"
        ),
        'user_template': (
            "Guide the student to the most relevant pages and sources for each question. Give a brief explanation for each, but do NOT provide direct answers. If any question is about a syllabus or schedule, you may answer directly.\n"
            "When referencing sources, smoothly cite them in-text, e.g. \"You should be able to find more about that on page 233.\" or \"To learn more about this, turn to page 253.\" List sources and context below.\n\n"
            "Sources List:\n<source_name> (Page <page_number>)\n\n"
            "Context:\n{context}\n\n"
            "Original Questions:\n{original_query}\n\n"
            "Processed Questions:\n{processed_queries}"
        )
    }
}

# Debug and Logging Configuration - settings for debugging output
ENABLE_DETAILED_LOGGING = False
SHOW_TEXT_PREVIEWS = True
PREVIEW_LENGTH = 200
SHOW_EMBEDDING_SAMPLES = True
EMBEDDING_SAMPLE_SIZE = 5

# Error Handling - retry settings for when things go wrong
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Memory Optimization - batch sizes to prevent memory issues
BATCH_EMBEDDING_SIZE = 50