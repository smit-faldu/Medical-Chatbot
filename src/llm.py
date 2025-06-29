"""
Gemini LLM Integration
Handles interaction with Google's Gemini AI model.
"""
import os
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiLLM:
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-04-17"):
        self.model_name = model_name
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
        )

    def create_medical_prompt(self, user_question: str, context: str, confidence: float, confidence_threshold: float = 0.3) -> str:
        system_prompt = f"""You are a medical knowledge assistant that provides accurate, specific answers based on the MedMCQA dataset context.

CRITICAL INSTRUCTIONS:
1. ALWAYS provide the SPECIFIC, CORRECT answer based on the medical context
2. If the question asks about a specific condition, mechanism, or outcome, give the exact answer
3. Do NOT give general explanations when a specific answer is available
4. Use the provided context as your primary source of information
5. Be precise and factual - if you know the answer, state it directly
6. Only use fallback if the context is truly irrelevant or empty

CONTEXT PROVIDED FROM MedMCQA DATASET:
{context}

CONFIDENCE SCORE: {confidence:.3f}
USER QUESTION: {user_question}

RESPONSE FORMAT - Return ONLY valid JSON in this exact structure:

If confidence < {confidence_threshold}:
{{
  "answer": "I'm not confident enough to answer that based on my current knowledge base.",
  "explanation": "The retrieved context doesn't provide sufficient information to answer this question confidently.",
  "key_points": ["Low confidence score", "Insufficient context"],
  "subject": "General"
}}

If confidence >= {confidence_threshold}:
{{
  "answer": "[SPECIFIC, DIRECT ANSWER - e.g., 'Atrophy', 'ACE inhibitors', 'Myocardial infarction']",
  "explanation": "[Detailed explanation of why this specific answer is correct, including pathophysiology, mechanisms, and medical reasoning based on the context]",
  "key_points": ["Key medical concept 1", "Key medical concept 2", "Clinical significance"],
  "subject": "[Medical field from context like Pathology, Pharmacology, etc.]"
}}

ANSWER REQUIREMENTS:
- If the question asks "what leads to X" or "what causes Y", provide the specific outcome/result
- If the question asks about a specific condition, provide the exact condition name
- If the question asks about mechanisms, provide the specific mechanism
- Be direct and specific - avoid vague or general answers when specific information is available

CRITICAL FORMATTING RULES:
1. Return ONLY a valid JSON object
2. NO markdown code blocks (no ```)
3. NO additional text before or after the JSON
4. NO explanations outside the JSON
5. Start your response with {{ and end with }}

REQUIRED JSON FORMAT:
{{
  "answer": "Specific answer here",
  "explanation": "Detailed explanation here",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "subject": "Medical field name"
}}

Your response must be valid JSON that can be parsed directly."""
        
        return system_prompt

    def generate_response(self, user_question: str, context: str, confidence: float, confidence_threshold: float = 0.3) -> str:
        try:
            prompt = self.create_medical_prompt(user_question, context, confidence, confidence_threshold)
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            answer_text = response.text.strip()
            return answer_text
        except Exception as e:
            return '{"answer": "I am not confident enough to answer that based on my current knowledge base.", "explanation": "An error occurred while generating the response.", "key_points": ["API error", "Please try again"], "subject": "Error"}'

    def _extract_subject_from_context(self, context: str) -> str:
        if "Subject:" in context:
            lines = context.split('\n')
            for line in lines:
                if "Subject:" in line:
                    subject = line.split("Subject:")[1].split(",")[0].strip()
                    return subject
        return "Medical Knowledge" 