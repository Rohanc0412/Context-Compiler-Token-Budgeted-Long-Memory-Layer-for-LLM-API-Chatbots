#!/usr/bin/env bash
set -euo pipefail

USER="u_demo"
SESSION="s_demo_1"
BASE_URL="http://localhost:8000"
HEADER="Content-Type: application/json"

say_chat () {
  local msg="$1"
  echo
  echo "User message: $msg"
  curl -sS -X POST "$BASE_URL/chat" -H "$HEADER" \
    -d "{\"user_id\":\"$USER\",\"session_id\":\"$SESSION\",\"message\":\"$msg\"}"
  echo
}

say_compile () {
  local msg="$1"
  echo
  echo "Compile for message: $msg"
  curl -sS -X POST "$BASE_URL/compile" -H "$HEADER" \
    -d "{\"user_id\":\"$USER\",\"session_id\":\"$SESSION\",\"message\":\"$msg\"}"
  echo
}

export_memory () {
  echo
  echo "Export memory for user and session"
  # If your export uses query params
  curl -sS "$BASE_URL/memory/export?user_id=$USER&session_id=$SESSION"
  echo
}

delete_memory () {
  echo
  echo "Delete memory for a clean run"
  curl -sS -X POST "$BASE_URL/memory/delete" -H "$HEADER" \
    -d "{\"user_id\":\"$USER\",\"session_id\":\"$SESSION\"}"
  echo
}

run_summary () {
  echo
  echo "Run summarizer"
  curl -sS -X POST "$BASE_URL/summaries/run" -H "$HEADER" \
    -d "{\"user_id\":\"$USER\",\"session_id\":\"$SESSION\"}"
  echo
}

echo "Using user_id=$USER session_id=$SESSION"

# Step 0: clean slate
delete_memory

# Step 1: seed constraint and preference in one sentence
say_chat "I must avoid peanuts and I prefer vegan meals."

# Step 2: seed a task with a deadline
say_chat "Task: finish the quarterly report by Friday."

# Step 3: seed a decision
say_chat "We decided to use provider X for embeddings."

# Step 4: send filler that should NOT become memories
say_chat "Thanks!"
say_chat "Ok cool"
say_chat "How are you?"

# Step 5: inspect compiled prompt BEFORE running summarizer
# What to look for in compiled_prompt:
# - active_constraints includes avoid peanuts
# - preferences includes vegan meals (may be in a preferences section or in rolling summary if you do that)
# - task_state includes quarterly report task
# - rolling_summary might be empty or minimal before summarizer runs
say_compile "Suggest a dinner idea."

# Step 6: export memory to verify stored items are clean and concise
# What to look for:
# - constraint value should be concise: avoid peanuts
# - preference value should be concise: vegan meals
# - task value should be finish the quarterly report
# - decision value should be use provider X for embeddings
# - status should be active
# - source_event_ids should be UUIDs
export_memory

# Step 7: run summarizer and inspect compiled prompt again
run_summary

# If your summarizer runs async, add a short wait
sleep 2

# What to look for in compiled_prompt now:
# - rolling_summary populated with constraints, preferences, open_tasks, decisions
say_compile "Summarize what you know about me and my current tasks."

# Step 8: contradiction update to test conflict handling
# What to look for:
# - old constraint about peanuts should become outdated
# - new memory should be stored and active
say_chat "Update: peanuts are okay now."

# Step 9: run summarizer again to see it reflect the updated memory state
run_summary
sleep 2

# Step 10: export memory again and verify outdated marking worked
export_memory

# Step 11: compile again and ensure active constraints reflect the newest state
say_compile "Can you recommend a meal plan for tonight?"

echo
echo "Done. Check outputs for clean memory extraction and rolling summary behavior."
