#!/usr/bin/env bash
# ── 端到端冒烟测试 ──────────────────────────────────────────
# 验证所有关键路径在真实环境中能跑通。
# 用法: bash scripts/smoke_test.sh
set -e

GREEN='\033[92m'
RED='\033[91m'
CYAN='\033[96m'
RESET='\033[0m'
PASS="${GREEN}✓${RESET}"
FAIL="${RED}✗${RESET}"

echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}  RAG Document Q&A — Smoke Test${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

ERRORS=0

# 1. Unit tests
echo -e "[1/6] Running unit tests..."
if pytest tests/ -q --tb=short 2>&1 | tail -3; then
    echo -e "  ${PASS} Unit tests passed"
else
    echo -e "  ${FAIL} Unit tests failed"
    ERRORS=$((ERRORS + 1))
fi

# 2. Syntax check all Python files
echo -e "\n[2/6] Syntax checking Python files..."
FILES="app.py demo.py $(find src/ eval/ scripts/ -name '*.py' 2>/dev/null)"
SYNTAX_OK=0
SYNTAX_FAIL=0
for f in $FILES; do
    if python -m py_compile "$f" 2>/dev/null; then
        SYNTAX_OK=$((SYNTAX_OK + 1))
    else
        echo -e "  ${FAIL} $f"
        SYNTAX_FAIL=$((SYNTAX_FAIL + 1))
    fi
done
echo -e "  ${PASS} ${SYNTAX_OK} files OK, ${SYNTAX_FAIL} failed"
[ "$SYNTAX_FAIL" -gt 0 ] && ERRORS=$((ERRORS + 1))

# 3. Retrieval-only benchmark
echo -e "\n[3/6] Running retrieval-only benchmark..."
if python -m eval.benchmark --retrieval-only --configs baseline hybrid -v 2>&1 | tail -10; then
    echo -e "  ${PASS} Retrieval benchmark completed"
else
    echo -e "  ${FAIL} Retrieval benchmark failed"
    ERRORS=$((ERRORS + 1))
fi

# 4. Digital Self indexing
echo -e "\n[4/6] Testing Digital Self indexing..."
if python -c "
from src.ingestion.conversation_loader import ConversationLoader
loader = ConversationLoader(strategy='turn_group', turns_per_chunk=4, overlap_turns=1)
chunks = loader.load_and_chunk('data/conversations/tech_preferences.json')
print(f'  Loaded {len(chunks)} chunks from tech_preferences.json')
for c in chunks:
    print(f'    [{c.chunk_id}] {c.text[:60]}...')
print('  OK')
" 2>&1; then
    echo -e "  ${PASS} Conversation loader works"
else
    echo -e "  ${FAIL} Conversation loader failed"
    ERRORS=$((ERRORS + 1))
fi

# 5. Chinese tokenization
echo -e "\n[5/6] Testing Chinese tokenization..."
if python -c "
from src.retrieval.sparse_retriever import BM25Retriever
r = BM25Retriever()
tokens = r._tokenize('自然语言处理是人工智能的重要方向')
print(f'  Tokens: {tokens}')
assert len(tokens) > 2, 'Expected more than 2 tokens for Chinese text'
print('  OK')
" 2>&1; then
    echo -e "  ${PASS} Chinese tokenization works"
else
    echo -e "  ${FAIL} Chinese tokenization failed"
    ERRORS=$((ERRORS + 1))
fi

# 6. LLM client import check
echo -e "\n[6/6] Verifying LLM client imports..."
if python -c "
from src.generation.llm_client import AnthropicClient, OpenAIClient, OllamaClient
print('  AnthropicClient, OpenAIClient, OllamaClient — all importable')
print('  OK')
" 2>&1; then
    echo -e "  ${PASS} LLM clients importable"
else
    echo -e "  ${FAIL} LLM client import failed"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}  All checks passed!${RESET}"
else
    echo -e "${RED}  ${ERRORS} check(s) failed${RESET}"
fi
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

exit $ERRORS
