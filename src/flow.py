"""
LangGraph Medical Chat Flow
Orchestrates the conversation flow using LangGraph.
"""
import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .pinecone_retriever import MedMCQAPineconeRetriever
from .llm import GeminiLLM

class ChatState(TypedDict):
    user_question: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    confidence: float
    llm_response: Dict[str, Any]
    final_answer: str
    should_fallback: bool
    error: str

class MedicalChatFlow:
    def __init__(self, data_dir: str = "data", confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.retriever = MedMCQAPineconeRetriever(data_dir=data_dir)
        self.llm = GeminiLLM()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        workflow.add_node("user_input", self.process_user_input)
        workflow.add_node("retriever", self.retrieve_context)
        workflow.add_node("confidence_check", self.check_confidence)
        workflow.add_node("llm_generate", self.generate_llm_response)
        workflow.add_node("format_response", self.format_final_response)
        workflow.add_node("fallback_response", self.create_fallback_response)
        workflow.set_entry_point("user_input")
        workflow.add_edge("user_input", "retriever")
        workflow.add_edge("retriever", "confidence_check")
        workflow.add_conditional_edges(
            "confidence_check",
            self.should_use_llm,
            {"generate": "llm_generate", "fallback": "fallback_response"}
        )
        workflow.add_edge("llm_generate", "format_response")
        workflow.add_edge("fallback_response", "format_response")
        workflow.add_edge("format_response", END)
        return workflow.compile()

    def process_user_input(self, state: ChatState) -> ChatState:
        user_question = state.get("user_question", "").strip()
        if not user_question:
            state["error"] = "Empty question provided"
            state["should_fallback"] = True
        return state

    def retrieve_context(self, state: ChatState) -> ChatState:
        try:
            user_question = state["user_question"]
            relevant_docs, confidence = self.retriever.get_relevant_context(user_question, k=3, min_score=0.2)
            context = self.retriever.format_context_for_llm(relevant_docs)
            state["retrieved_docs"] = relevant_docs
            state["context"] = context
            state["confidence"] = confidence
        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["confidence"] = 0.0
            state["context"] = ""
            state["retrieved_docs"] = []
        return state

    def check_confidence(self, state: ChatState) -> ChatState:
        confidence = state.get("confidence", 0.0)
        has_error = bool(state.get("error"))
        context = state.get("context", "").strip()
        should_fallback = (
            confidence < self.confidence_threshold or 
            has_error or 
            not context
        )
        state["should_fallback"] = should_fallback
        return state

    def should_use_llm(self, state: ChatState) -> str:
        return "fallback" if state.get("should_fallback", False) else "generate"

    def generate_llm_response(self, state: ChatState) -> ChatState:
        try:
            user_question = state["user_question"]
            context = state["context"]
            confidence = state["confidence"]
            llm_response_text = self.llm.generate_response(user_question, context, confidence, self.confidence_threshold)
            import json
            import re
            try:
                clean_response = self._clean_json_response(llm_response_text)
                llm_response = json.loads(clean_response)
                required_fields = ["answer", "explanation", "key_points", "subject"]
                for field in required_fields:
                    if field not in llm_response:
                        raise ValueError(f"Missing required field: {field}")
                llm_response["confidence"] = confidence
                llm_response["source"] = "MedMCQA"
                llm_response["is_fallback"] = False
            except (json.JSONDecodeError, ValueError):
                answer_text = self._extract_answer_from_text(llm_response_text)
                llm_response = {
                    "answer": answer_text,
                    "explanation": "Response generated successfully but not in structured JSON format.",
                    "key_points": ["See answer above for details"],
                    "subject": "General",
                    "confidence": confidence,
                    "source": "MedMCQA",
                    "is_fallback": False
                }
            state["llm_response"] = llm_response
        except Exception as e:
            state["error"] = f"LLM generation error: {str(e)}"
            state["should_fallback"] = True
        return state

    def create_fallback_response(self, state: ChatState) -> ChatState:
        fallback_message = "I'm not confident enough to answer that based on my current knowledge base."
        state["llm_response"] = {
            "answer": fallback_message,
            "explanation": "The retrieved context doesn't provide sufficient information to answer this question confidently.",
            "key_points": ["Low confidence score", "Insufficient context"],
            "subject": "General",
            "confidence": state.get("confidence", 0.0),
            "source": "Fallback",
            "is_fallback": True
        }
        return state

    def format_final_response(self, state: ChatState) -> ChatState:
        llm_response = state.get("llm_response", {})
        if not llm_response or not llm_response.get("answer"):
            final_response = {
                "question": state["user_question"],
                "answer": "I'm not confident enough to answer that based on my current knowledge base.",
                "explanation": "No valid response was generated.",
                "key_points": ["System error", "Please try again"],
                "subject": "Error",
                "confidence": 0.0,
                "source": "Error",
                "is_fallback": True
            }
        else:
            final_response = {
                "question": state["user_question"],
                "answer": llm_response.get("answer", "Unable to generate response"),
                "explanation": llm_response.get("explanation", "No explanation available"),
                "key_points": llm_response.get("key_points", []),
                "subject": llm_response.get("subject", "General"),
                "confidence": llm_response.get("confidence", 0.0),
                "source": llm_response.get("source", "Unknown"),
                "is_fallback": llm_response.get("is_fallback", True)
            }
        state["final_answer"] = final_response
        return state

    def _clean_json_response(self, response_text: str) -> str:
        import re
        clean_text = response_text.strip()
        clean_text = re.sub(r'^```json\s*\n?', '', clean_text, flags=re.IGNORECASE | re.MULTILINE)
        clean_text = re.sub(r'^```\s*\n?', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r'\n?```\s*$', '', clean_text, flags=re.MULTILINE)
        first_brace = clean_text.find('{')
        if first_brace != -1:
            clean_text = clean_text[first_brace:]
        last_brace = clean_text.rfind('}')
        if last_brace != -1:
            clean_text = clean_text[:last_brace + 1]
        clean_text = clean_text.strip()
        return clean_text

    def _extract_answer_from_text(self, text: str) -> str:
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('```', '{', '}')):
                if line.lower().startswith('answer:'):
                    return line.split(':', 1)[1].strip()
                elif line.lower().startswith('**answer:**'):
                    return line.split('**answer:**', 1)[1].strip()
                elif len(line) > 10:
                    return line
        return text.strip()

    def chat(self, user_question: str) -> Dict[str, Any]:
        initial_state = ChatState(
            user_question=user_question,
            retrieved_docs=[],
            context="",
            confidence=0.0,
            llm_response={},
            final_answer="",
            should_fallback=False,
            error=""
        )
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state["final_answer"]
        except Exception as e:
            return {
                "question": user_question,
                "answer": "I'm not confident enough to answer that based on my current knowledge base.",
                "confidence": 0.0,
                "source": "Error",
                "is_fallback": True,
                "error": str(e)
            }

    def get_flow_stats(self) -> Dict[str, Any]:
        return {
            "retriever_stats": self.retriever.get_stats(),
            "confidence_threshold": self.confidence_threshold,
            "llm_model": self.llm.model_name
        } 