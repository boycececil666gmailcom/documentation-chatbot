# region App Setup
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import vector_db
from .agent_flow import agent_graph
from .config import (
    ALLOW_CREDENTIALS,
    ALLOWED_ORIGINS,
    BACKEND_HOST,
    BACKEND_PORT,
    OPENROUTER_MODEL,
)
from .models import QueryRequest, QueryResponse

app = FastAPI(title="Theme-Based RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)
# endregion


# region Query Endpoints
@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """Executes the agent workflow graph for user queries."""
    try:
        inputs = {
            "query": request.query,
            "history": [msg.model_dump() for msg in request.history],
            "attempt_count": 0,
        }
        result = await agent_graph.ainvoke(inputs)
        tools_used = ["retrieve_VDB"] if result.get("retrieved_documents") else []

        return QueryResponse(
            response=result.get("final_response", ""),
            citations=result.get("citations", []),
            tool_calls_executed=tools_used,
            should_hyde=result.get("should_hyde"),
            hyde_reason=result.get("hyde_reason"),
            hyde_content=result.get("hyde_content"),
            retrieved_documents=result.get("retrieved_documents"),
            history=result.get("history"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}") from e


@app.get("/health")
async def health_check():
    """Returns backend and vector store health status."""
    try:
        vector_db.get_vector_store("raptor_chunks")
        vector_ok = "ok"
    except Exception:
        vector_ok = "degraded"

    return {
        "status": "ok",
        "model": OPENROUTER_MODEL,
        "platform": "Theme-Based RAG Workflow",
        "vector_store": vector_ok,
    }


# endregion

# region Server Runner
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
    )
# endregion
