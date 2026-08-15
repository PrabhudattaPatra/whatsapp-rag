"""
Custom action overrides for the NeMo Guardrails config in this directory.

NeMo's ActionDispatcher loads actions in order (built-ins -> library ->
CWD -> this config directory, loaded LAST via dict.update()) -- see
settings.guardrails_config_path ("./config") in src/config.py. A function
defined here with the same name as a built-in action cleanly overrides it,
with no extra registration wiring needed anywhere else.

Why self_check_input needs an override:
    NeMo's built-in self_check_input action (nemoguardrails/library/
    self_check/input_check/actions.py) judges topic-relevance from the
    CURRENT user message alone. src/graph/nodes/agent.py already passes the
    full conversation history into RunnableRails, but NeMo's own master
    flow runs input rails on $user_message (this turn's isolated text)
    BEFORE creating the UserMessage event that would update
    context["last_user_message"] -- so by the time this action runs, that
    context var still holds the *previous* turn, and the current turn is
    evaluated with zero awareness of it. A short, context-dependent
    follow-up like "give me for btech" (after "what is the fee structure
    for Mtech") has nothing to anchor it to the university, and gets
    misjudged as off-topic.

    Colang's context replay keeps exactly one prior turn around for free --
    context["last_user_message"] / context["last_bot_message"] -- so this
    override just threads that into the self_check_input prompt (see the
    `history` block added to that task in config/prompts.yml) instead of
    discarding it. Everything else about the action is unchanged from the
    built-in implementation.
"""

import logging
from typing import Optional

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.actions.llm.utils import llm_call, warn_if_truncated
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.types import LLMModel
from nemoguardrails.utils import new_event_dict

log = logging.getLogger(__name__)


def _build_recent_history(context: dict, current_user_input: str) -> str:
    """Format the immediately preceding user/bot turn, if any.

    Colang's replay-based context only reliably retains ONE prior turn
    (last_user_message/last_bot_message get overwritten on every replayed
    event) -- that's exactly enough to resolve short follow-ups that depend
    on the immediately preceding exchange. Returns "" on the first turn of
    a conversation, or defensively if last_user_message somehow already
    equals the current input (would mean the current turn's context wasn't
    isolated the way we expect).
    """
    prev_user = context.get("last_user_message")
    prev_bot = context.get("last_bot_message")

    if not prev_user or prev_user == current_user_input:
        return ""

    lines = [f"user: {prev_user}"]
    if prev_bot:
        lines.append(f"assistant: {prev_bot}")
    return "\n".join(lines)


@action(is_system_action=True)
async def self_check_input(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[LLMModel] = None,
    config: Optional[RailsConfig] = None,
    **kwargs,
):
    """CVRGU override of NeMo's built-in self_check_input action.

    Identical to the built-in implementation, except the rendered prompt
    also receives a `history` variable (see _build_recent_history above),
    so short context-dependent follow-ups are judged in light of the prior
    on-topic exchange instead of in total isolation.
    """
    _MAX_TOKENS = 1024
    context = context or {}
    user_input = context.get("user_message")
    task = Task.SELF_CHECK_INPUT

    if user_input:
        history = _build_recent_history(context, user_input)

        prompt = llm_task_manager.render_task_prompt(
            task=task,
            context={
                "user_input": user_input,
                "history": history,
            },
        )
        stop = llm_task_manager.get_stop_tokens(task=task)
        max_tokens = llm_task_manager.get_max_tokens(task=task)
        max_tokens = max_tokens or _MAX_TOKENS

        llm_call_info_var.set(LLMCallInfo(task=task.value))

        llm_response = await llm_call(
            llm,
            prompt,
            stop=stop,
            llm_params={
                "temperature": config.lowest_temperature,
                "max_tokens": max_tokens,
            },
        )
        warn_if_truncated(llm_response, task.value)
        response = llm_response.content

        log.info(f"Input self-checking result is: `{response}`.")

        if llm_task_manager.has_output_parser(task):
            result = llm_task_manager.parse_task_output(task, output=response)
        else:
            result = llm_task_manager.parse_task_output(
                task, output=response, forced_output_parser="is_content_safe"
            )

        is_safe = result[0]

        if not is_safe:
            return ActionResult(
                return_value=False,
                events=[new_event_dict("mask_prev_user_message", intent="unanswerable message")],
            )

        return is_safe
