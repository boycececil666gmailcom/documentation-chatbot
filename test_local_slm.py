# region LocalSLMTest
import sys
import time
import urllib.request
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import LOCAL_SLM_BASE_URL, LOCAL_SLM_MODEL
from src.llm_client import local_slm

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    print(f"[LocalSLM-main] Target Endpoint: {LOCAL_SLM_BASE_URL}")
    print(f"[LocalSLM-main] Target Model:    {LOCAL_SLM_MODEL}")

    print(f"[LocalSLM-main] Testing reachability at {LOCAL_SLM_BASE_URL}...")
    try:
        # Check base health/root endpoint
        health_url = LOCAL_SLM_BASE_URL.replace("/v1", "/health") if ":8000" in LOCAL_SLM_BASE_URL else LOCAL_SLM_BASE_URL.replace("/v1", "")
        urllib.request.urlopen(health_url, timeout=2.0)
        print("[LocalSLM-main] Endpoint is reachable and online.")
    except Exception as ping_exc:
        print(f"[LocalSLM-main] Endpoint {LOCAL_SLM_BASE_URL} is currently offline: {ping_exc}")
        print("[LocalSLM-main] Start the server using 'docker compose up -d' or './boot.sh'.")
        return

    system_prompt = (
        "Extract only key technical terms, identifiers, API names, and core keywords "
        "from the user query for sparse keyword search.\n"
        "Remove conversational filler, questions, and stop words.\n"
        "Output ONLY space-separated keywords without punctuation or quotes."
    )
    user_query = "Kanziで3Dノードのプロパティを変更するC++ APIコードを教えてください"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    print(f"[LocalSLM-main] Sending test query: '{user_query}'")
    start_time = time.perf_counter()
    try:
        response = local_slm.invoke(messages, timeout=10.0)
        duration = time.perf_counter() - start_time
        print(f"[LocalSLM-main] Extracted Keywords: '{response.content.strip()}'")
        print(f"[LocalSLM-main] Elapsed Time:       {duration:.3f}s")
        print("[LocalSLM-main] Test SUCCESSFUL.")
    except Exception as exc:
        duration = time.perf_counter() - start_time
        print(f"[LocalSLM-main] Test FAILED ({duration:.3f}s): {exc}")


if __name__ == "__main__":
    main()
# endregion

